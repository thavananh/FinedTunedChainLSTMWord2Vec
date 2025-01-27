# package.py
import os


class PackageInstaller:
    def __init__(self, packages):
        # Lưu danh sách các gói cần cài
        self.packages = packages

    def install_packages(self):
        # Cài đặt từng gói trong danh sách
        for package in self.packages:
            os.system(f'pip install {package}')

    def add_package(self, package):
        # Thêm gói mới vào danh sách
        self.packages.append(package)

    def remove_package(self, package):
        # Xóa gói khỏi danh sách
        if package in self.packages:
            self.packages.remove(package)

    def list_packages(self):
        # In ra danh sách các gói hiện tại
        print(f'Packages: {os.system("pip list")}')


