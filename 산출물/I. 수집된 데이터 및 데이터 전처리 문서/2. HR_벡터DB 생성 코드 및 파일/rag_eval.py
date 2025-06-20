import json
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from evaluate import load

# 모델 로딩
def load_reranker_model(path="work1/models/bge-reranker-v2-m3-ko"):
    return CrossEncoder(path, device="cuda")

def load_vectorstore(folder="faiss_seq", index_name="index", embedding=None):
    return FAISS.load_local(folder_path=folder, embeddings=embedding, index_name=index_name, allow_dangerous_deserialization=True)

def load_eval_questions(path="eval_questions_merged.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Rerank
def rerank_documents(query, docs, reranker, top_k=1):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc.page_content for doc, _ in reranked[:top_k]]

# GPT 호출
def generate_answer(query, context, client, prompt_style="cot"):
    if prompt_style == "cot":
        system_prompt = (
            "너는 회사 규정을 안내하는 전문 QA 비서야.\n"
            "답변 전에 문서 내용을 한 단계씩 정리하고, 논리적으로 답을 도출해.\n"
            "문서에 명확한 근거가 없으면 '문서에 없습니다'라고 답해."
        )
    else:
        system_prompt = "당신은 회사 규정에 대해 정확하게 답변하는 도우미입니다."

    user_prompt = f"""문서:
{context}

질문: {query}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 평가
def evaluate_answer(prediction, reference, qid):
    f1 = load("evaluate-metric/squad", "f1")
    em = load("evaluate-metric/squad", "exact_match")

    pred_dict = {"id": qid, "prediction_text": prediction}
    ref_dict = {"id": qid, "answers": [{"text": reference, "answer_start": 0}]}

    f1_score = f1.compute(predictions=[pred_dict], references=[ref_dict])["f1"]
    em_score = em.compute(predictions=[pred_dict], references=[ref_dict])["exact_match"]
    return f1_score, em_score

# 전체 평가 루프
def run_full_evaluation(client, embedding, reranker_path="work1/models/bge-reranker-v2-m3-ko", top_k=1, prompt_style="cot"):
    reranker = load_reranker_model(reranker_path)
    vectorstore = load_vectorstore(embedding=embedding)
    questions = load_eval_questions()

    all_f1, all_em = [], []

    for idx, qa in enumerate(questions):
        qid = qa.get("id", str(idx))
        query = qa["question"]
        answer = qa["answer"]

        docs = vectorstore.similarity_search(query, k=5)
        top_context = "\n\n".join(rerank_documents(query, docs, reranker, top_k=top_k))

        prediction = generate_answer(query, top_context, client, prompt_style=prompt_style)
        f1, em = evaluate_answer(prediction, answer, qid)

        all_f1.append(f1)
        all_em.append(em)

    print(f"📊 평균 F1: {sum(all_f1)/len(all_f1):.2f}")
    print(f"📊 평균 EM: {sum(all_em)/len(all_em):.2f}")
    

def generate_answer(query, context, client, prompt_style="cot"):
    if prompt_style == "basic":
        system_prompt = "당신은 회사 규정에 대해 정확하게 답변하는 도우미입니다."
    elif prompt_style == "strict":
        system_prompt = (
            "너는 회사의 사내 규정을 정확히 안내하는 QA 비서야. "
            "문서에 기반해서만 답변하고, 없으면 '문서에 없습니다'라고 답해."
        )
    elif prompt_style == "cot":
        system_prompt = (
            "너는 회사 규정을 안내하는 QA 비서야. "
            "문서를 정리하고, 논리적으로 생각한 뒤 답변해. "
            "근거 없으면 '문서에 없습니다'라고 해."
        )
    elif prompt_style == "step":
        system_prompt = (
            "문서를 기반으로 답변 전 단계를 나눠서 설명한 뒤, 결론을 내려주세요."
        )
    else:
        raise ValueError(f"알 수 없는 프롬프트 스타일: {prompt_style}")

    user_prompt = f"""문서:
{context}

질문: {query}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
  
def run_prompt_ab_test(client, embedding, prompt_styles, top_k=1):
    from rag_eval import run_full_evaluation

    for style_key in prompt_styles:
        print(f"\n🔍 실험 프롬프트: {style_key}")
        run_full_evaluation(
            client=client,
            embedding=embedding,
            top_k=top_k,
            prompt_style=style_key
        )

def run_custom_prompt_eval(client, embedding, reranker_path="work1/models/bge-reranker-v2-m3-ko", top_k=1, system_prompt=""):
    from sentence_transformers import CrossEncoder
    from langchain_community.vectorstores import FAISS
    from evaluate import load
    import json
    from tqdm import tqdm

    # 벡터DB 및 리랭커 로드
    reranker = CrossEncoder(reranker_path, device="cuda")
    vectorstore = FAISS.load_local(
        folder_path="faiss_seq",
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )

    # 질문 로드
    with open("eval_questions_merged.jsonl", "r", encoding="utf-8") as f:
        eval_questions = [json.loads(line) for line in f]

    f1 = load("evaluate-metric/squad", "f1")
    em = load("evaluate-metric/squad", "exact_match")

    f1_scores, em_scores = [], []

    for idx, qa in tqdm(enumerate(eval_questions), total=len(eval_questions)):
        query = qa["question"]
        answer = qa["answer"]
        qid = qa.get("id", str(idx))

        docs = vectorstore.similarity_search(query, k=top_k)
        reranker_inputs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(reranker_inputs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        top_context = "\n\n".join([doc.page_content for doc, _ in reranked[:top_k]])

        user_prompt = f"""문서:
{top_context}

질문: {query}"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        prediction = completion.choices[0].message.content.strip()

        prediction_dict = {"id": qid, "prediction_text": prediction}
        reference_dict = {
            "id": qid,
            "answers": [{"text": answer, "answer_start": 0}]
        }

        f1_score = f1.compute(predictions=[prediction_dict], references=[reference_dict])["f1"]
        em_score = em.compute(predictions=[prediction_dict], references=[reference_dict])["exact_match"]

        f1_scores.append(f1_score)
        em_scores.append(em_score)

    print(f"📊 평균 F1: {sum(f1_scores)/len(f1_scores):.2f}")
    print(f"📊 평균 EM: {sum(em_scores)/len(em_scores):.2f}")

def run_custom_prompt_eval_win(client, embedding, eval_path, reranker_path="work1/models/bge-reranker-v2-m3-ko", top_k=1, system_prompt=""):
    from sentence_transformers import CrossEncoder
    from langchain_community.vectorstores import FAISS
    from evaluate import load
    import json
    from tqdm import tqdm

    # 벡터DB 및 리랭커 로드
    reranker = CrossEncoder(reranker_path, device="cuda")
    vectorstore = FAISS.load_local(
        folder_path="faiss_win",
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )

    # 🔹 평가 질문 로드 (매개변수로 전달된 경로 사용)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_questions = [json.loads(line) for line in f]

    f1 = load("evaluate-metric/squad", "f1")
    em = load("evaluate-metric/squad", "exact_match")

    f1_scores, em_scores = [], []

    for idx, qa in tqdm(enumerate(eval_questions), total=len(eval_questions)):
        query = qa["question"]
        answer = qa["answer"]
        qid = qa.get("id", str(idx))

        docs = vectorstore.similarity_search(query, k=top_k)
        reranker_inputs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(reranker_inputs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        top_context = "\n\n".join([doc.page_content for doc, _ in reranked[:top_k]])

        user_prompt = f"""문서:
{top_context}

질문: {query}"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        prediction = completion.choices[0].message.content.strip()

        prediction_dict = {"id": qid, "prediction_text": prediction}
        reference_dict = {
            "id": qid,
            "answers": [{"text": answer, "answer_start": 0}]
        }

        f1_score = f1.compute(predictions=[prediction_dict], references=[reference_dict])["f1"]
        em_score = em.compute(predictions=[prediction_dict], references=[reference_dict])["exact_match"]

        f1_scores.append(f1_score)
        em_scores.append(em_score)

    print(f"📊 평균 F1: {sum(f1_scores)/len(f1_scores):.2f}")
    print(f"📊 평균 EM: {sum(em_scores)/len(em_scores):.2f}")
