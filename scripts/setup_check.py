import sys

def check_python():
    print(f"🐍 Python: {sys.version}")
    if sys.version_info >= (3, 10):
        print("✅ Versão OK")
        return True
    print("❌ Python 3.10+ necessário")
    return False

def check_imports():
    packages = ['cv2', 'mediapipe', 'numpy', 'sklearn', 'pandas']
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} OK")
        except ImportError:
            print(f"❌ {pkg} não instalado")
            all_ok = False
    return all_ok

def check_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("✅ Câmera OK")
                return True
        print("❌ Câmera não acessível")
        return False
    except:
        print("❌ Erro ao acessar câmera")
        return False

def main():
    print("=" * 50)
    print("🔍 VERIFICAÇÃO DE SETUP")
    print("=" * 50)
    
    results = [
        check_python(),
        check_imports(),
        check_camera()
    ]
    
    print("=" * 50)
    if all(results):
        print("🎉 TUDO OK!")
    else:
        print("⚠️  Corrija os problemas acima")

if __name__ == "__main__":
    main()