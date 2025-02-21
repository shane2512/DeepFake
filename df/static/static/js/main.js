// Prevent default drag behaviors
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Handle drag over
function handleDragOver(e) {
  preventDefaults(e);
  e.target.classList.add("border-indigo-500");
}

// Handle drag leave
function handleDragLeave(e) {
  preventDefaults(e);
  e.target.classList.remove("border-indigo-500");
}

// Handle drop
function handleDrop(e) {
  preventDefaults(e);
  e.target.classList.remove("border-indigo-500");

  const dt = e.dataTransfer;
  const files = dt.files;

  if (files.length) {
    handleFiles(files);
  }
}

// Handle file selection from the hidden input
function handleFileSelect(input) {
  if (input.files.length) {
    handleFiles(input.files);
  }
}

// The actual upload with Fetch
function handleFiles(files) {
  const file = files[0]; // We'll handle just the first file
  if (!file) return;

  // Allowed MIME types
  const allowedTypes = [
    "image/png",
    "image/jpeg",
    "video/mp4",
    "video/avi",
    "video/quicktime",
  ];

  if (!allowedTypes.includes(file.type)) {
    alert("Please upload only PNG, JPG, MP4, AVI, or MOV files.");
    return;
  }

  // Show the loading spinner
  const loadingDiv = document.getElementById("loading");
  loadingDiv.classList.remove("hidden");

  // Prepare the form data
  const formData = new FormData();
  formData.append("file", file);

  // POST to Flask endpoint
  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Upload failed");
      }
      return response.text();
    })
    .then((html) => {
      // Replace the entire HTML with the server response
      document.documentElement.innerHTML = html;
      // Reinitialize Lucide icons
      if (typeof lucide !== "undefined") {
        lucide.createIcons();
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred during upload. Please try again.");
      loadingDiv.classList.add("hidden");
    });
}

// Initialize event listeners on DOMContentLoaded
document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    // Prevent default form submission
    uploadForm.addEventListener("submit", (e) => e.preventDefault());

    // Drag & drop events
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      uploadForm.addEventListener(eventName, preventDefaults, false);
    });

    // Click to open file dialog
    uploadForm.addEventListener("click", function (e) {
      if (e.target !== document.getElementById("fileInput")) {
        document.getElementById("fileInput").click();
      }
    });
  }
});
