# Ví dụ sử dụng:
from Package import PackageInstaller

if __name__ == "__main__":
    # Khởi tạo đối tượng với danh sách gói cần cài
    installer = PackageInstaller(['pyvi', 'underthesea'])

    # Cài đặt các gói
    installer.install_packages()

    # In ra danh sách gói đã cài
    print(installer.list_packages())


    