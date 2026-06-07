#!/usr/bin/env python3
"""
AI Tutor 애플리케이션 진입점

리팩토링된 구조:
- app/api/main.py: FastAPI 앱 및 라우터
- app/core/: 핵심 비즈니스 로직 (models, llm)
- app/services/: 서비스 레이어 (rag, audio, stt)
- app/utils/: 유틸리티 함수
- config/: JSON 설정 파일
"""

if __name__ == "__main__":
    # app.api.main의 서버 실행 코드를 직접 실행
    from app.api.main import app, config, logger
    import uvicorn
    from pathlib import Path

    # SSL 인증서 확인
    ssl_key = Path(config.SSL_KEYFILE)
    ssl_cert = Path(config.SSL_CERTFILE)
    use_ssl = ssl_key.exists() and ssl_cert.exists()

    # 서버 설정
    server_config = {
        "app": app,  # 앱 객체 직접 전달
        "host": config.HOST,
        "port": config.PORT,
        "reload": False,
        "log_level": "info"
    }

    if use_ssl:
        server_config["ssl_keyfile"] = str(ssl_key)
        server_config["ssl_certfile"] = str(ssl_cert)
        protocol = "https"
        logger.info(f"[HTTPS] 보안 서버 시작")
    else:
        protocol = "http"
        logger.warning("[HTTP] SSL 인증서를 찾을 수 없습니다")
        logger.warning("       음성 기능은 HTTPS에서만 작동합니다")

    logger.info(f"접속 주소: {protocol}://{config.HOST}:{config.PORT}")
    logger.info(f"로컬 접속: {protocol}://localhost:{config.PORT}")

    # 서버 시작
    uvicorn.run(**server_config)
