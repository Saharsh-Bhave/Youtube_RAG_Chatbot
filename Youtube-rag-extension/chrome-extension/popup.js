document.getElementById("askBtn").addEventListener("click", async () =>{
    const question = document.getElementById("question").value;
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true});

    const videoId = new URLSearchParams(new URL(tab.url).search).get("v");

    if(!videoId){
        document.getElementById("response").innerText = "Not a Valid YouTube video.";
        return;
    }

    document.getElementById("response").innerText = "Thinking...";

    try{
        const res = await fetch("https://youtube-rag-api.onrender.com/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question, video_id: videoId}),
        });

        const data = await res.json();
        document.getElementById("response").innerText= data.answer;
    } catch(err) {
        document.getElementById("response").innerText = "Error getting a response";
    }
});