// Upload PDF to backend
function uploadPDF() {
    let fileInput = document.getElementById("pdfUpload");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select a file!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("uploadStatus").innerText = data.message || data.error;
    })
    .catch(error => console.error("Error:", error));
}

function askQuestion() {
    let question = document.getElementById("questionInput").value.trim();
    let chatHistory = document.getElementById("chatHistory");

    if (!question) {
        alert("Please enter a question!");
        return;
    }

    // Display user message on the left
    let userMessage = document.createElement("div");
    userMessage.classList.add("self-start", "bg-blue-500", "text-white", "p-3", "rounded-lg", "mb-2", "max-w-xs");
    userMessage.innerText = "You: " + question;
    chatHistory.appendChild(userMessage);
    
    // Clear input field
    document.getElementById("questionInput").value = "";

    // Scroll to the latest message
    chatHistory.scrollTop = chatHistory.scrollHeight;

    fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        // Display AI response on the right
        let aiMessageDiv = document.createElement("div");
        aiMessageDiv.classList.add("flex","w-full", "justify-end") 
        
        let len = data.answer.length

        let aiMessage = document.createElement("p");
        aiMessage.classList.add("self-end", "bg-gray-300", "text-gray-800", "p-3", "rounded-lg", "mb-2","w-1/2");
        aiMessageDiv.appendChild(aiMessage);
        chatHistory.appendChild(aiMessageDiv);

        console.log("len", len);
        let answer =""
        for (let i = 0; i < len; i++) {
            setTimeout(() => {
                answer += data.answer[i];  // Append one character at a time
                aiMessage.innerText = "AI: " + answer;
            }, i * 10);  // Delay increases with each character
        }

        // Scroll to the latest message
        chatHistory.scrollTop = chatHistory.scrollHeight;
    })
    .catch(error => console.error("Error:", error));
}
