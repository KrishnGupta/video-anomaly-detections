function showFileName() {
    const fileInput = document.getElementById('file-upload');
    const fileName = fileInput.files[0].name;
    document.getElementById('file-name').textContent = `Selected file: ${fileName}`;
}
