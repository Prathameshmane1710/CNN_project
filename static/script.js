const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const uploadText = document.getElementById("uploadText");
const checkBtn = document.getElementById("checkBtn");
const result = document.getElementById("result");

fileInput.onchange = () => {
    const file = fileInput.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        uploadText.style.display = "none";
        result.textContent = "";
    }
};

checkBtn.onclick = async () => {
    const file = fileInput.files[0];
    if (!file) { alert("Please choose an image first."); return; }

    const formData = new FormData();
    formData.append("image", file);

    result.textContent = "Checking...";
    result.className = "";

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();

        const label = data.prediction;

        if (data.defective_probability === null) {
            result.className = "";
            result.textContent = label;
        } else {
            const prob = (data.defective_probability * 100).toFixed(1);
            const cssClass = label === "Defective" ? "defective" : "ok";
            result.className = cssClass;
            result.innerHTML = `${label}<span class="result-prob">defective probability: ${prob}%</span>`;
        }
    } catch (err) {
        result.textContent = "Something went wrong. Is the server running?";
    }
};