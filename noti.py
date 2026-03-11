import subprocess

def show_popup():
    script = '''
    display dialog "Đã xác nhận thành công!" with title "Thông báo" buttons {"OK"} default button "OK"
    '''
    subprocess.run(["osascript", "-e", script])