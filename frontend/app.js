const API = "https://cardiovision-agmq.onrender.com";

// Helper function to show loading state
function setLoading(isLoading, buttonId) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  
  if (isLoading) {
    button.disabled = true;
    button.style.opacity = "0.6";
    button.dataset.originalText = button.innerText;
    button.innerText = "Loading...";
  } else {
    button.disabled = false;
    button.style.opacity = "1";
    if (button.dataset.originalText) {
      button.innerText = button.dataset.originalText;
    }
  }
}

// Helper function to display messages
function showMessage(elementId, message, isError = true) {
  const msgElement = document.getElementById(elementId);
  if (msgElement) {
    msgElement.innerText = message;
    msgElement.style.color = isError ? "#d32f2f" : "#4caf50";
  }
}

// Check if user is authenticated
function checkAuth() {
  const token = localStorage.getItem("token");
  if (!token && !window.location.pathname.includes("login") && !window.location.pathname.includes("signup")) {
    window.location.href = "login.html";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const predictBtn = document.getElementById("predictBtn");
  const downloadBtn = document.getElementById("downloadBtn");

  if (predictBtn) {
    predictBtn.addEventListener("click", predict);
    checkAuth();
  }
  if (downloadBtn) downloadBtn.addEventListener("click", downloadReport);
});

// ---------------- LOGIN ----------------
function login() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;

  if (!email || !password) {
    showMessage("msg", "Please enter both email and password");
    return;
  }

  setLoading(true, "loginBtn");

  fetch(API + "/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  })
  .then(async res => {
    const data = await res.json();

    if (res.status === 404) {
      window.location.href = "signup.html?msg=no_account";
      return;
    }

    if (!res.ok) {
      showMessage("msg", data.detail || "Login failed");
      return;
    }

    localStorage.setItem("token", data.access_token);
    window.location.href = "dashboard.html";
  })
  .catch(err => {
    console.error("Login error:", err);
    showMessage("msg", "Connection error. Please check your internet.");
  })
  .finally(() => {
    setLoading(false, "loginBtn");
  });
}

// ---------------- SIGNUP ----------------
function signupUser() {
  const username = document.getElementById("username").value.trim();
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;

  if (!username || !email || !password) {
    showMessage("msg", "Please fill in all fields");
    return;
  }

  if (password.length < 8) {
    showMessage("msg", "Password must be at least 8 characters");
    return;
  }

  setLoading(true, "signupBtn");

  fetch(API + "/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password })
  })
  .then(async res => {
    const data = await res.json();
    
    if (!res.ok) {
      showMessage("msg", data.detail || "Signup failed");
      return;
    }
    
    showMessage("msg", "Account created! Redirecting to login...", false);
    setTimeout(() => {
      window.location.href = "login.html";
    }, 2000);
  })
  .catch(err => {
    console.error("Signup error:", err);
    showMessage("msg", "Connection error. Please try again.");
  })
  .finally(() => {
    setLoading(false, "signupBtn");
  });
}

function goToSignup() {
  window.location.href = "signup.html";
}

// ---------------- PREDICT ----------------
function predict() {
  const token = localStorage.getItem("token");
  if (!token) {
    alert("Session expired. Please login again.");
    window.location.href = "login.html";
    return;
  }

  const fileInput = document.getElementById("imageFile");
  if (!fileInput.files.length) {
    alert("Please select an image first");
    return;
  }

  const file = fileInput.files[0];
  const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/bmp"];
  
  if (!validTypes.includes(file.type)) {
    alert("Invalid file type. Please upload a JPG, PNG, or BMP image.");
    return;
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    alert("File too large. Maximum size is 10MB.");
    return;
  }

  const img = document.getElementById("heatmap");
  const riskText = document.getElementById("riskText");

  // Reset UI
  img.style.display = "none";
  img.classList.remove("show");
  riskText.innerText = "Analyzing image...";
  riskText.style.color = "#0a6ebd";

  setLoading(true, "predictBtn");

  const form = new FormData();
  form.append("file", file);

  fetch(API + "/predict", {
    method: "POST",
    headers: {
      Authorization: "Bearer " + token
    },
    body: form
  })
  .then(async res => {
    if (res.status === 401) {
      alert("Session expired. Please login again.");
      window.location.href = "login.html";
      return;
    }
    
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Prediction failed");
    }
    
    return res.json();
  })
  .then(data => {
    if (!data) return;

    console.log("📊 Prediction Response:", data);
    console.log("🔗 Heatmap URL:", data.heatmap_url);

    //Image loading with debugging
    img.onerror = function(e) {
      console.error("❌ Image failed to load");
      console.error("Failed URL:", data.heatmap_url);
      console.error("Error:", e);
      
      riskText.innerHTML = "⚠️ Result generated but image failed to load.<br>Make sure you're accessing via http://127.0.0.1:3000";
      riskText.style.color = "red";
      
      alert("Image load failed!\n\nMake sure you're serving the frontend properly:\n1. Open new terminal\n2. cd to frontend folder\n3. Run: python -m http.server 3000\n4. Access: http://127.0.0.1:3000/login.html");
    };

    img.onload = function() {
      console.log("✅ Image loaded successfully!");
      img.style.display = "block";
      setTimeout(() => {
        img.classList.add("show");
      }, 50);
    };

    // Set image source with cache busting
    const timestamp = new Date().getTime();
    img.src = data.heatmap_url + "?t=" + timestamp;
    
    console.log("🖼️ Image src set to:", img.src);

    // Display risk
    if (data.risk === "High Risk") {
      riskText.innerHTML = "⚠️ <strong>HEART DISEASE DETECTED</strong><br>Contact a cardiologist ASAP.<br>Confidence: " + data.confidence + "%";
      riskText.style.color = "red";
    }
    else if (data.risk === "Moderate Risk") {
      riskText.innerHTML = "⚡ <strong>Moderate Risk Detected</strong><br>Take care of your heart health.<br>Confidence: " + data.confidence + "%";
      riskText.style.color = "orange";
    }
    else {
      riskText.innerHTML = "✅ <strong>Healthy Heart Detected</strong><br>No immediate risk found.<br>Confidence: " + data.confidence + "%";
      riskText.style.color = "green";
    }
  })
  .catch(err => {
    console.error("❌ Prediction error:", err);
    riskText.innerText = "";
    alert("Prediction failed: " + err.message);
  })
  .finally(() => {
    setLoading(false, "predictBtn");
  });
}

// ---------------- REPORT ----------------
function downloadReport() {
  const token = localStorage.getItem("token");
  
  if (!token) {
    alert("Session expired. Please login again.");
    window.location.href = "login.html";
    return;
  }

  setLoading(true, "downloadBtn");

  fetch(API + "/download-report", {
    headers: { Authorization: "Bearer " + token }
  })
  .then(async res => {
    if (res.status === 401) {
      alert("Session expired. Please login again.");
      window.location.href = "login.html";
      return;
    }

    if (res.status === 404) {
      alert("No analysis found. Please analyze an image first.");
      return;
    }

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Report download failed");
    }

    return res.blob();
  })
  .then(blob => {
    if (!blob) return;

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "CardioVision_Report.pdf";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log("✅ Report downloaded successfully");
  })
  .catch(err => {
    console.error("Download error:", err);
    alert("Report download failed: " + err.message);
  })
  .finally(() => {
    setLoading(false, "downloadBtn");
  });
}

// Logout function
function logout() {
  localStorage.removeItem("token");
  window.location.href = "login.html";
}