CV Match AI

CV Match AI is a Telegram bot powered by a Large Language Model (LLM), designed to help students and junior specialists tailor their resumes (CVs) to specific job descriptions (JDs).

The project analyzes the "fit" between a resume and a job opening, provides actionable feedback for improvement, and helps candidates pass through Applicant Tracking Systems (ATS), significantly increasing their chances of getting an interview.

(It is highly recommended to add a GIF or screenshot of the bot in action here.)
Key Features

CV to JD Fit Analysis: Evaluates how well a resume matches a job description, providing a percentage score.

Personalized Recommendations: Offers specific advice on what skills to add, what to rephrase, or what to remove.

Similar Job Matching: Finds and suggests up to 5 relevant job openings from a local database based on the CV.

File Format Support: Processes .pdf files, .docx files, and plain text.

Tech Stack

Python 3.10+

python-telegram-bot library for interacting with the Telegram API.

Ollama for local LLM execution (utilizing the llama3:8b model).

Pandas for handling the job dataset.

pypdf and python-docx for parsing files.

Getting Started

To run the bot locally on your machine, follow these steps.

1. Prerequisites

Ensure you have the following installed:

Python (version 3.10 or higher)

Git

Ollama

After installing Ollama, pull the required model by running the following command in your terminal:

ollama pull llama3:8b

2. Installation

Clone the repository:

git clone https://github.com/TimurIshmurzinDS/CVAI.git

cd CVAI


3. Configuration

Configure the Bot Token:

In the root directory of the project, create a file named .env.

Open the file and add a single line, replacing YOUR_TOKEN_HERE with your actual Telegram bot token:

TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"

This file is listed in .gitignore, so your token will not be committed to GitHub.

Prepare the Dataset:

For the job matching feature to work, the bot expects a file named cleaned_jobs.csv in the root directory.

This file must contain at least the following columns: title, company, location, and SearchText.


4. Running the Application

Start the Ollama server:

Ensure the Ollama application is running on your machine.

Run the bot:

From the root project directory, with your virtual environment activated, execute the following command:

python bot.py

Your bot is now running and ready to be used in Telegram.

Project Structure

.

├── bot.py             # The main script containing the bot's logic

├── cleaned_jobs.csv   # The processed dataset of job listings (must be added)

├── .env               # Stores secret tokens (created locally)

├── .gitignore         # Specifies files for Git to ignore

└── README.md          # This file
