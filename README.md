# Youtube_RAG_Chatbot
RAG based chatbot to ask questions about any youtube video.

Also contains the extension for the same chatbot that can be run locally. 

Steps to run the extension locally on your system:
1. Clone the repository
2. Make sure you run `pip install -r requirementsf.txt` to install all the necessary libraries.
3. Now in your terminal go to the "backend" directory.
4. Type the following command `uvicorn app:app --reload`. This will start the ASGI (web server) for our implementation.
5. Now in Google Chrome's extension menu turn on developer mode and select "load unpacked". Select the "chrome-extension" folder.
6. Use YouTube in your chrome browser and ask away!!
