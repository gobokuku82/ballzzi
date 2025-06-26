<<<<<<< HEAD
// 회사 소개 슬라이더 변수
let currentSlideIndex = 0;
let slides = [];
let totalSlides = 0;
let slidesWrapper = null;
let indicators = [];

// 고객 슬라이더 변수
let currentCustomerSlide = 0;
const totalCustomerSlides = 5;
let autoSlideInterval;

// 회사 소개 슬라이더 함수들
function showSlide(index) {
    if (!slidesWrapper || totalSlides === 0) return;
    
    if (index >= totalSlides) currentSlideIndex = 0;
    if (index < 0) currentSlideIndex = totalSlides - 1;
    
    slidesWrapper.style.transform = `translateX(-${currentSlideIndex * 100}%)`;
    
    // 인디케이터 업데이트
    indicators.forEach((indicator, i) => {
        indicator.classList.toggle('active', i === currentSlideIndex);
    });
    
    // 버튼 상태 업데이트
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (prevBtn) prevBtn.disabled = currentSlideIndex === 0;
    if (nextBtn) nextBtn.disabled = currentSlideIndex === totalSlides - 1;
}

function changeSlide(direction) {
    currentSlideIndex += direction;
    showSlide(currentSlideIndex);
}

function currentSlide(index) {
    currentSlideIndex = index - 1;
    showSlide(currentSlideIndex);
}

// 고객 슬라이더 함수들
function updateCustomerSlider() {
    const customerSliderWrapper = document.getElementById('customerSliderWrapper');
    if (!customerSliderWrapper) return;
    
    const translateX = -currentCustomerSlide * 100;
    customerSliderWrapper.style.transform = `translateX(${translateX}%)`;
    
    // 인디케이터 업데이트
    const customerIndicators = document.querySelectorAll('.slider-indicators .indicator');
    customerIndicators.forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentCustomerSlide);
    });
}

function nextSlide() {
    currentCustomerSlide = (currentCustomerSlide + 1) % totalCustomerSlides;
    updateCustomerSlider();
    resetAutoSlide();
}

function prevSlide() {
    currentCustomerSlide = (currentCustomerSlide - 1 + totalCustomerSlides) % totalCustomerSlides;
    updateCustomerSlider();
    resetAutoSlide();
}

function goToSlide(slideIndex) {
    currentCustomerSlide = slideIndex;
    updateCustomerSlider();
    resetAutoSlide();
}

function startAutoSlide() {
    autoSlideInterval = setInterval(() => {
        nextSlide();
    }, 10000);
}

function resetAutoSlide() {
    clearInterval(autoSlideInterval);
    startAutoSlide();
}
=======
// 🚨 JavaScript 파일 로딩 테스트 (즉시 실행)
console.log('🚨🚨🚨 JAVASCRIPT 파일 로드 확인! 🚨🚨🚨');
console.log('현재 시간:', new Date().toLocaleTimeString());
>>>>>>> 9c306d4c8bda5295a555921b696d78a479aebdb8

// DOM이 로드된 후 실행
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚨 DOM 로드 완료! ballzzi 시스템 시작!');
    
    // 회사 소개 슬라이더 초기화
    slides = document.querySelectorAll('.intro-slide');
    totalSlides = slides.length;
    slidesWrapper = document.getElementById('slidesWrapper');
    indicators = document.querySelectorAll('.slide-indicators .indicator');
    
    if (slidesWrapper && totalSlides > 0) {
        showSlide(0);
    }
    
    // 고객 슬라이더 초기화
    const customerSliderWrapper = document.getElementById('customerSliderWrapper');
    if (customerSliderWrapper) {
        updateCustomerSlider();
        startAutoSlide();
        
        // 마우스 호버시 자동 슬라이드 일시정지
        const sliderContainer = document.querySelector('.customers-section .slider-container');
        if (sliderContainer) {
            sliderContainer.addEventListener('mouseenter', () => {
                clearInterval(autoSlideInterval);
            });

            sliderContainer.addEventListener('mouseleave', () => {
                startAutoSlide();
            });
        }
        
        // 키보드 네비게이션
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                prevSlide();
            } else if (e.key === 'ArrowRight') {
                nextSlide();
            }
        });
    }
    
    // 햄버거 메뉴 토글
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }
    
    // 네비게이션 링크 클릭 시 메뉴 닫기
    document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
        if (hamburger && navMenu) {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        }
    }));
    
    // 스크롤 시 네비게이션 배경 변경
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(255, 255, 255, 0.98)';
                navbar.style.boxShadow = '0 5px 30px rgba(0, 0, 0, 0.15)';
            } else {
                navbar.style.background = 'rgba(255, 255, 255, 0.95)';
                navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
            }
        }
    });
    
    // 부드러운 스크롤을 위한 함수
    function smoothScroll(target, duration) {
        const targetElement = document.querySelector(target);
        if (!targetElement) return;
        
        const targetPosition = targetElement.offsetTop - 70; // 네비게이션 높이만큼 조정
        const startPosition = window.pageYOffset;
        const distance = targetPosition - startPosition;
        let startTime = null;
        
        function animation(currentTime) {
            if (startTime === null) startTime = currentTime;
            const timeElapsed = currentTime - startTime;
            const run = ease(timeElapsed, startPosition, distance, duration);
            window.scrollTo(0, run);
            if (timeElapsed < duration) requestAnimationFrame(animation);
        }
        
        function ease(t, b, c, d) {
            t /= d / 2;
            if (t < 1) return c / 2 * t * t + b;
            t--;
            return -c / 2 * (t * (t - 2) - 1) + b;
        }
        
        requestAnimationFrame(animation);
    }
    
    // 네비게이션 링크에 부드러운 스크롤 적용
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('href');
            if (target !== '#' && document.querySelector(target)) {
                smoothScroll(target, 1000);
            }
        });
    });
    
    // 스크롤 애니메이션을 위한 Intersection Observer
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // 애니메이션할 요소들 관찰
    const animateElements = document.querySelectorAll('.service-card, .feature-card, .video-container, .about-content, .contact-item');
    animateElements.forEach(el => {
        observer.observe(el);
    });
    
    // 스크롤 인디케이터 애니메이션
    const scrollIndicator = document.querySelector('.scroll-indicator');
    if (scrollIndicator) {
        scrollIndicator.addEventListener('click', function() {
            smoothScroll('#video', 1000);
        });
    }
    
    // CTA 버튼 클릭 이벤트
    const ctaButton = document.querySelector('.cta-button');
    if (ctaButton) {
        ctaButton.addEventListener('click', function() {
            window.location.href = '/chatbot/';
        });
    }
    
    // 서비스 및 특징 버튼 클릭 이벤트
    document.querySelectorAll('.service-btn, .feature-btn').forEach(button => {
        button.addEventListener('click', function() {
            window.location.href = '/chatbot/';
        });
    });
    
    // 네비게이션 활성 상태 업데이트
    function updateActiveNav() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }
    
    // 스크롤 시 활성 네비게이션 업데이트
    window.addEventListener('scroll', updateActiveNav);
    
    // 페이지 로드 시 초기 활성 네비게이션 설정
    updateActiveNav();
    
    // 통계 숫자 카운터 애니메이션
    function animateCounter(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            element.innerHTML = Math.floor(progress * (end - start) + start) + (element.dataset.suffix || '');
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
    
    // 통계 섹션이 보일 때 카운터 애니메이션 시작
    const statsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const statNumbers = entry.target.querySelectorAll('.stat-item h4');
                statNumbers.forEach(stat => {
                    const text = stat.textContent;
                    const number = parseInt(text.replace(/\D/g, ''));
                    const suffix = text.replace(/\d/g, '');
                    stat.dataset.suffix = suffix;
                    stat.textContent = '0' + suffix;
                    animateCounter(stat, 0, number, 2000);
                });
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    const statsSection = document.querySelector('.company-stats');
    if (statsSection) {
        statsObserver.observe(statsSection);
    }
    
    // 페이지 스크롤 진행 표시바 (선택사항)
    function updateScrollProgress() {
        const scrollTop = window.scrollY;
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        
        // 프로그레스 바가 있다면 업데이트
        const progressBar = document.querySelector('.scroll-progress');
        if (progressBar) {
            progressBar.style.width = scrollPercent + '%';
        }
    }
    
    window.addEventListener('scroll', updateScrollProgress);
    
    // 플로팅 챗봇 기능
    const floatingChatbot = document.getElementById('floating-chatbot');
    const heroLogo = document.querySelector('.hero-logo');
    
    if (floatingChatbot) {
        // 플로팅 챗봇 클릭 이벤트
        floatingChatbot.addEventListener('click', function() {
            window.location.href = '/chatbot/';
        });
        
        // 스크롤 시 히어로 로고와 플로팅 챗봇 전환
        window.addEventListener('scroll', function() {
            const heroSection = document.getElementById('home');
            if (heroSection && heroLogo) {
                const heroBottom = heroSection.offsetTop + heroSection.offsetHeight;
                const scrollPosition = window.scrollY + window.innerHeight * 0.3;
                
                if (scrollPosition > heroBottom) {
                    // 히어로 섹션을 벗어나면 히어로 로고 숨기고 플로팅 챗봇 표시
                    heroLogo.classList.add('hidden');
                    floatingChatbot.style.opacity = '1';
                    floatingChatbot.style.transform = 'scale(1)';
                } else {
                    // 히어로 섹션 내에서는 히어로 로고 표시하고 플로팅 챗봇 숨기기
                    heroLogo.classList.remove('hidden');
                    floatingChatbot.style.opacity = '0';
                    floatingChatbot.style.transform = 'scale(0.8)';
                }
            }
        });
        
        // 초기 상태 설정
        floatingChatbot.style.opacity = '0';
        floatingChatbot.style.transform = 'scale(0.8)';
    }
    
    // 모바일에서 터치 제스처 개선
    let touchStartY = 0;
    document.addEventListener('touchstart', function(e) {
        touchStartY = e.touches[0].clientY;
    }, { passive: true });
    
    document.addEventListener('touchend', function(e) {
        const touchEndY = e.changedTouches[0].clientY;
        const diff = touchStartY - touchEndY;
        
        // 위로 스와이프 감지 (최소 50px)
        if (diff > 50) {
            // 다음 섹션으로 스크롤하는 로직을 추가할 수 있습니다
        }
    }, { passive: true });
    
    // 페이지 로드 완료 후 애니메이션 시작
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
        
        // 히어로 섹션 애니메이션
        const heroContent = document.querySelector('.hero-content');
        if (heroContent) {
            heroContent.style.animation = 'fadeInUp 1s ease-out';
        }
    });
    
    // 키보드 네비게이션 지원
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // ESC 키로 모든 모달 닫기
            if (hamburger && navMenu) {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            }
        }
    });
    
    // 접근성 개선: 포커스 관리
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });
    
    // Django static 경로 자동 감지
    const getStaticPath = (filename) => {
        const staticBase = document.querySelector('img[src*="ballzzi.png"]')?.src.replace('ballzzi.png', '') || '/static/icons/';
        return staticBase + filename;
    };

    // 다양한 감정의 ballzzi 이미지와 멘트 배열 (공통 사용)
    const ballzziStates = [
        {
            image: getStaticPath('ballzzi.png'),
            message: '안녕하세요! Ballzzi입니다 🏆'
        },
        {
            image: getStaticPath('shy_ballzzi.png'),
            message: '아...아니에요... 부끄러워요 😳'
        },
        {
            image: getStaticPath('angry_ballzzi.png'),
            message: '으아악! 화나요! 😤'
        },
        {
            image: getStaticPath('evil_ballzzi.png'),
            message: '후후후... 내 계획이 완벽해! 😈'
        },
        {
            image: getStaticPath('sad_ballzzi.png'),
            message: '흑흑... 슬퍼요... 😢'
        },
        {
            image: getStaticPath('shame_ballzzi.png'),
            message: '아... 민망해요... 🙈'
        }
    ];

    console.log('🎯 Static 경로 감지 결과:', ballzziStates[0].image);

    // 🚨 강력한 디버깅 모드 활성화
    console.log('=== 🔍 BALLZZI 디버깅 시작 ===');
    console.log('1. 모든 ballzzi 상태:', ballzziStates);
    
    // 모든 가능한 요소 검색
    console.log('2. 페이지의 모든 관련 요소들:');
    console.log('- nav-logo-hover:', document.getElementById('nav-logo-hover'));
    console.log('- logo-image:', document.getElementById('logo-image'));
    console.log('- tooltip-message:', document.getElementById('tooltip-message'));
    console.log('- hero-logo-hover:', document.getElementById('hero-logo-hover'));
    console.log('- hero-logo-image:', document.getElementById('hero-logo-image'));
    console.log('- hero-tooltip-message:', document.getElementById('hero-tooltip-message'));

    // 네비게이션 로고 hover 효과 - 이미지와 멘트 변경
    const navLogoHover = document.getElementById('nav-logo-hover');
    const navLogoImage = document.getElementById('logo-image');
    const navTooltipMessage = document.getElementById('tooltip-message');
    
    console.log('3. 네비게이션 로고 요소들:', {
        navLogoHover: navLogoHover,
        navLogoImage: navLogoImage,
        navTooltipMessage: navTooltipMessage
    });
    
    if (navLogoHover && navLogoImage && navTooltipMessage) {
        console.log('네비게이션 로고 hover 효과 초기화 성공!');
        
        let navCurrentStateIndex = 0;
        let isNavHovering = false;
        
        // 네비게이션 로고에 마우스 올릴 때마다 **랜덤** 상태로 변경
        navLogoHover.addEventListener('mouseenter', function() {
            isNavHovering = true;
            console.log('네비게이션 로고 hover 이벤트 발생!');
            
            // 🎲 랜덤 상태 선택 (현재 상태 제외)
            let randomIndex;
            do {
                randomIndex = Math.floor(Math.random() * ballzziStates.length);
            } while (randomIndex === navCurrentStateIndex && ballzziStates.length > 1);
            
            navCurrentStateIndex = randomIndex;
            const currentState = ballzziStates[navCurrentStateIndex];
            
            console.log('네비게이션 로고 랜덤 상태 변경:', {
                randomIndex: randomIndex,
                emotion: currentState.message,
                image: currentState.image
            });
            
            // 이미지와 메시지 변경
            navLogoImage.src = currentState.image;
            navTooltipMessage.textContent = currentState.message;
        });
        
        // 마우스가 벗어날 때 기본 상태로 복귀
        navLogoHover.addEventListener('mouseleave', function() {
            isNavHovering = false;
            setTimeout(() => {
                if (!isNavHovering) {
                    // 기본 상태로 복귀
                    navLogoImage.src = ballzziStates[0].image;
                    navTooltipMessage.textContent = ballzziStates[0].message;
                    navCurrentStateIndex = 0;
                    console.log('네비게이션 로고 기본 상태로 복귀');
                }
            }, 500); // 0.5초 후 복귀
        });
        
    } else {
        console.log('네비게이션 로고 요소를 찾을 수 없습니다!');
    }

    // 🎯 HERO 로고 전용 디버깅 및 기능
    console.log('=== 🎯 HERO 로고 디버깅 시작 ===');
    
    const heroLogoHover = document.getElementById('hero-logo-hover');
    const heroLogoImage = document.getElementById('hero-logo-image');
    const heroTooltipMessage = document.getElementById('hero-tooltip-message');
    
    console.log('HERO 요소 검색 결과:');
    console.log('- heroLogoHover:', heroLogoHover);
    console.log('- heroLogoImage:', heroLogoImage);
    console.log('- heroTooltipMessage:', heroTooltipMessage);
    
    // 모든 이미지 요소 검색 (혹시 ID가 다를 수 있음)
    const allImages = document.querySelectorAll('img');
    console.log('페이지의 모든 이미지 요소들:');
    allImages.forEach((img, index) => {
        if (img.src.includes('ballzzi')) {
            console.log(`${index}: ID="${img.id}", class="${img.className}", src="${img.src}"`);
        }
    });
    
    if (heroLogoHover && heroLogoImage && heroTooltipMessage) {
        console.log('✅ 히어로 로고 모든 요소 발견! 이벤트 리스너 등록 시작...');
        
        let heroCurrentStateIndex = 0;
        let isHeroHovering = false;
        
        // 🔥 즉시 테스트용 클릭 이벤트도 추가 (디버깅용)
        heroLogoHover.addEventListener('click', function() {
            console.log('🔥 HERO 로고 클릭 테스트!');
            
            // 랜덤 상태 선택
            let randomIndex = Math.floor(Math.random() * ballzziStates.length);
            const currentState = ballzziStates[randomIndex];
            
            console.log('클릭 테스트 상태 변경:', {
                randomIndex: randomIndex,
                emotion: currentState.message,
                image: currentState.image
            });
            
            // 이미지와 메시지 강제 변경
            heroLogoImage.src = currentState.image;
            heroTooltipMessage.textContent = currentState.message;
            
            // 이미지 로딩 확인
            heroLogoImage.onload = function() {
                console.log('✅ 이미지 로딩 성공:', this.src);
            };
            
            heroLogoImage.onerror = function() {
                console.error('❌ 이미지 로딩 실패:', this.src);
            };
        });
        
        // 히어로 로고에 마우스 올릴 때마다 **랜덤** 상태로 변경
        heroLogoHover.addEventListener('mouseenter', function() {
            isHeroHovering = true;
            console.log('🎯 히어로 로고 HOVER 이벤트 발생!');
            
            // 🎲 랜덤 상태 선택 (현재 상태 제외)
            let randomIndex;
            do {
                randomIndex = Math.floor(Math.random() * ballzziStates.length);
            } while (randomIndex === heroCurrentStateIndex && ballzziStates.length > 1);
            
            heroCurrentStateIndex = randomIndex;
            const currentState = ballzziStates[heroCurrentStateIndex];
            
            console.log('🎲 히어로 로고 랜덤 상태 변경:', {
                randomIndex: randomIndex,
                emotion: currentState.message,
                image: currentState.image
            });
            
            // 이미지와 메시지 변경
            heroLogoImage.src = currentState.image;
            heroTooltipMessage.textContent = currentState.message;
            
            // 이미지 로딩 상태 확인
            heroLogoImage.onload = function() {
                console.log('✅ HOVER 이미지 로딩 성공:', this.src);
            };
            
            heroLogoImage.onerror = function() {
                console.error('❌ HOVER 이미지 로딩 실패:', this.src);
            };
        });
        
        // 마우스가 벗어날 때 기본 상태로 복귀
        heroLogoHover.addEventListener('mouseleave', function() {
            isHeroHovering = false;
            console.log('🚪 히어로 로고 mouseleave 이벤트');
            
            setTimeout(() => {
                if (!isHeroHovering) {
                    console.log('🔄 히어로 로고 기본 상태로 복귀 시작...');
                    // 기본 상태로 복귀
                    heroLogoImage.src = ballzziStates[0].image;
                    heroTooltipMessage.textContent = ballzziStates[0].message;
                    heroCurrentStateIndex = 0;
                    console.log('✅ 히어로 로고 기본 상태로 복귀 완료');
                }
            }, 500); // 0.5초 후 복귀
        });
        
    } else {
        console.error('❌ 히어로 로고 요소를 찾을 수 없습니다!');
        console.log('누락된 요소:');
        if (!heroLogoHover) console.log('- hero-logo-hover 없음');
        if (!heroLogoImage) console.log('- hero-logo-image 없음');
        if (!heroTooltipMessage) console.log('- hero-tooltip-message 없음');
    }
    
});

// CSS 애니메이션 클래스 추가
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        animation: slideInUp 0.8s ease-out forwards;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .nav-link.active::after {
        width: 100% !important;
    }
    
    .hamburger.active .bar:nth-child(2) {
        opacity: 0;
    }
    
    .hamburger.active .bar:nth-child(1) {
        transform: translateY(8px) rotate(45deg);
    }
    
    .hamburger.active .bar:nth-child(3) {
        transform: translateY(-8px) rotate(-45deg);
    }
    
    .keyboard-navigation *:focus {
        outline: 2px solid #667eea !important;
        outline-offset: 2px !important;
    }
    
    .loaded .hero-content {
        animation: fadeInUp 1s ease-out;
    }
    
    /* 스크롤 프로그레스 바 (선택사항) */
    .scroll-progress {
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        z-index: 9999;
        transition: width 0.1s ease;
    }
`;
document.head.appendChild(style); 
document.addEventListener('DOMContentLoaded', () => {
    const videoSection = document.querySelector('#video');
    const video = document.querySelector('#intro-video');

    // 페이지에 해당 요소들이 없으면 오류 방지를 위해 실행 중지
    if (!videoSection || !video) {
        return; 
    }

    // Intersection Observer 생성
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // 섹션이 화면에 보일 때
                video.currentTime = 0; // 영상을 처음부터 다시 재생
                video.play().catch(error => {
                    console.error("비디오 자동 재생 실패:", error);
                });
            } else {
                // 섹션이 화면에서 벗어날 때
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // 섹션이 50% 이상 보일 때 반응
    });

    // 비디오 섹션 관찰 시작
    observer.observe(videoSection);
});