import os
import subprocess
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile
import shutil

'''
LTAI5tC3Fe6uHAK92UcbUU7u
Q1zgpa4PTzqqjSv9qk1OcYZkp1hOy4
'''

app = Flask(__name__)
#app.secret_key = 'your-secret-key-here'  # 请更改为安全的密钥

# 配置
BUCKET_NAME = 'my-data11'
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'zip', 'rar'}

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_ossutil_command(command):
    """执行 ossutil 命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def parse_ls_output(output):
    """解析 ossutil ls 命令的输出"""
    lines = output.strip().split('\n')
    files = []
    folders = []

    for line in lines:
        # 跳过标题行、统计行和空行
        if ('LastModifiedTime' in line or 'Object Number is:' in line or
                'elapsed' in line or not line.strip()):
            continue

        # 尝试解析文件信息行
        parts = line.split()
        if len(parts) >= 6:
            try:
                # 标准格式：日期 时间 时区 大小 存储类型 ETAG 对象名称
                # 例如：2024-11-26 14:35:29 +0800 CST 12 Standard 1103F650EB2C292D179A032D2A97B0F5 oss://bucket/file.txt

                # 提取日期时间（前4个部分）
                date_time = ' '.join(parts[0:4])  # "2024-11-26 14:35:29 +0800 CST"
                size = parts[4]  # 文件大小
                storage_class = parts[5]  # 存储类型
                etag = parts[6]  # ETAG
                object_name = ' '.join(parts[7:])  # 完整的对象路径

                # 提取相对路径
                if object_name.startswith(f'oss://{BUCKET_NAME}/'):
                    file_path = object_name.replace(f'oss://{BUCKET_NAME}/', '')
                else:
                    continue  # 跳过不匹配的行

                if file_path.endswith('/'):
                    # 文件夹
                    folder_name = file_path.rstrip('/')
                    if folder_name:  # 避免空文件夹名
                        folders.append({
                            'name': os.path.basename(folder_name),
                            'path': file_path,
                            'type': 'folder'
                        })
                else:
                    # 文件
                    files.append({
                        'name': os.path.basename(file_path),
                        'path': file_path,
                        'size': size,
                        'modified': date_time,
                        'type': 'file'
                    })
            except (IndexError, ValueError):
                # 如果解析失败，跳过这一行
                continue

    return folders, files


@app.route('/')
def index():
    """主页 - 显示根目录文件"""
    return browse_folder('')


@app.route('/browse')
@app.route('/browse/<path:folder_path>')
def browse_folder(folder_path=''):
    """浏览文件夹"""
    # 构建 ossutil 命令
    oss_path = f'oss://{BUCKET_NAME}/'
    if folder_path:
        oss_path += folder_path + '/'

    command = f'ossutil ls {oss_path}'
    stdout, stderr, returncode = run_ossutil_command(command)

    folders = []
    files = []

    if returncode == 0:
        folders, files = parse_ls_output(stdout)
    else:
        flash(f'获取文件列表失败: {stderr}', 'error')

    # 构建面包屑导航
    breadcrumbs = []
    if folder_path:
        parts = folder_path.split('/')
        current_path = ''
        for part in parts:
            if part:
                current_path += part + '/'
                breadcrumbs.append({
                    'name': part,
                    'path': current_path.rstrip('/')
                })

    return render_template('index.html',
                           folders=folders,
                           files=files,
                           current_path=folder_path,
                           breadcrumbs=breadcrumbs,
                           bucket_name=BUCKET_NAME)


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有选择文件'})

    file = request.files['file']
    current_path = request.form.get('current_path', '')

    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # 保存到临时文件夹
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # 构建 OSS 路径
        oss_path = f'oss://{BUCKET_NAME}/'
        if current_path:
            oss_path += current_path + '/'
        oss_path += filename

        # 上传到 OSS
        command = f'ossutil cp "{temp_path}" {oss_path}'
        stdout, stderr, returncode = run_ossutil_command(command)

        # 删除临时文件
        os.remove(temp_path)

        if returncode == 0:
            return jsonify({'success': True, 'message': f'文件 {filename} 上传成功'})
        else:
            return jsonify({'success': False, 'message': f'上传失败: {stderr}'})

    return jsonify({'success': False, 'message': '不支持的文件类型'})


@app.route('/download/<path:file_path>')
def download_file(file_path):
    """下载文件"""
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    filename = os.path.basename(file_path)
    temp_file_path = os.path.join(temp_dir, filename)

    # 从 OSS 下载文件
    oss_path = f'oss://{BUCKET_NAME}/{file_path}'
    command = f'ossutil cp {oss_path} "{temp_file_path}"'
    stdout, stderr, returncode = run_ossutil_command(command)

    if returncode == 0:
        def remove_temp_file(response):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            return response

        return send_file(temp_file_path,
                         as_attachment=True,
                         download_name=filename)
    else:
        shutil.rmtree(temp_dir)
        flash(f'下载失败: {stderr}', 'error')
        return redirect(request.referrer or url_for('index'))


@app.route('/delete/<path:file_path>')
def delete_file(file_path):
    """删除文件"""
    oss_path = f'oss://{BUCKET_NAME}/{file_path}'
    command = f'ossutil rm {oss_path}'
    stdout, stderr, returncode = run_ossutil_command(command)

    if returncode == 0:
        flash(f'文件删除成功', 'success')
    else:
        flash(f'删除失败: {stderr}', 'error')

    return redirect(request.referrer or url_for('index'))


@app.route('/create_folder', methods=['POST'])
def create_folder():
    """创建文件夹"""
    folder_name = request.form.get('folder_name', '').strip()
    current_path = request.form.get('current_path', '')

    if not folder_name:
        return jsonify({'success': False, 'message': '文件夹名称不能为空'})

    # 构建 OSS 路径
    oss_path = f'oss://{BUCKET_NAME}/'
    if current_path:
        oss_path += current_path + '/'
    oss_path += folder_name + '/'

    # 创建文件夹（通过创建一个空对象）
    command = f'ossutil mkdir {oss_path}'
    stdout, stderr, returncode = run_ossutil_command(command)

    if returncode == 0:
        return jsonify({'success': True, 'message': f'文件夹 {folder_name} 创建成功'})
    else:
        return jsonify({'success': False, 'message': f'创建失败: {stderr}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)