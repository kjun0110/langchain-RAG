"""CPU/GPU 사용 확인 스크립트."""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import torch

    print("=" * 60)
    print("디바이스 사용 확인")
    print("=" * 60)

    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"\n[1] CUDA 사용 가능: {cuda_available}")

    if cuda_available:
        print(f"[2] CUDA 디바이스 개수: {torch.cuda.device_count()}")
        print(f"[3] 현재 CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"[4] CUDA 버전: {torch.version.cuda}")

        # 메모리 정보
        print(f"\n[5] GPU 메모리 정보:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    할당된 메모리: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"    예약된 메모리: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"    최대 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("[2] CPU 모드로 실행됩니다.")

    # Midm 모델이 로드되어 있는지 확인
    print(f"\n[6] Midm 모델 디바이스 확인:")
    try:
        from api.app.model.midm import load_midm_model
        from dotenv import load_dotenv
        import os

        load_dotenv()
        local_model_dir = os.getenv("LOCAL_MODEL_DIR")

        if local_model_dir:
            from pathlib import Path
            if not Path(local_model_dir).is_absolute():
                project_root = Path(__file__).parent
                local_model_dir = str(project_root / local_model_dir)

        print(f"  모델 경로: {local_model_dir or '기본값'}")
        print(f"  device_map 설정: auto (GPU가 있으면 자동 사용)")

        # 모델이 이미 로드되어 있는지 확인
        try:
            model = load_midm_model(
                model_path=local_model_dir if local_model_dir else None,
                register=False,
                is_default=False
            )

            if model.model:
                device = str(model.model.device) if hasattr(model.model, 'device') else "unknown"
                print(f"  실제 사용 디바이스: {device}")

                # 모델 정보 확인
                model_info = model.get_model_info()
                print(f"  모델 정보: {model_info}")
            else:
                print("  모델이 로드되지 않았습니다.")
        except Exception as e:
            print(f"  모델 로드 실패: {str(e)[:100]}")

    except ImportError as e:
        print(f"  모델 로더를 가져올 수 없습니다: {e}")

    print("\n" + "=" * 60)
    if cuda_available:
        print("[결론] GPU를 사용할 수 있습니다. device_map='auto'로 설정되어 있어")
        print("       GPU가 있으면 자동으로 GPU를 사용합니다.")
    else:
        print("[결론] CPU 모드로 실행됩니다. (GPU가 없거나 CUDA가 설치되지 않음)")
    print("=" * 60)

except ImportError:
    print("[오류] torch가 설치되지 않았습니다.")
    sys.exit(1)
except Exception as e:
    print(f"[오류] 오류 발생: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

