<p align="center"><h1 align="center">Resume Analyser for Job Description </h1></p>

<p align="center">
	<!-- local repository, no metadata badges. -->
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=default&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/spaCy-09A3D5.svg?style=default&logo=spaCy&logoColor=white" alt="spaCy">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
</p>
<br>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**ResumeAnalyser** is an automated resume ranking and analysis application designed for recruiters and hiring managers. This tool accepts a ZIP file containing multiple PDF resumes and an optional job description text file. It then processes the resumes using advanced NLP techniques and scoring algorithms to rank candidates based on:

- **Technical & Managerial Skills:** Evaluated using years of experience, education level, and keyword-based skills extraction.
- **Resume Quality:** Assessed via spell-check ratios, section identification, and overall brevity.
- **Job Matching:** Uses TF-IDF similarity measures to compare resumes against the provided job description.

The final output is a ranked list of resumes with scores and downloadable results for further review.

---

## Features

- **Interactive UI with Streamlit:** Upload resumes and job descriptions, view processing status, and download results.
- **PDF Text Extraction:** Converts resumes in PDF format into text using libraries like PyPDF2 and pymupdf.
- **NLP-Powered Analysis:** 
  - **Preprocessing:** Tokenization and lemmatization using spaCy.
  - **Skill Extraction:** Identifies both general and technical skills.
  - **Experience & Education:** Extracts years of experience and detects education level.
  - **Resume Quality Metrics:** Spell-check, section identification, and brevity evaluation.
- **Scoring and Ranking:** Combines technical, managerial, and overall quality scores with job match scores (via TF-IDF similarity) to generate final rankings.
- **Downloadable Results:** 
  - CSV file containing all calculated scores.
  - Top 3 ranked resumes available as downloadable PDFs with an embedded viewer.
- **Default Job Skills Management:** Automatically loads or creates a default job skills file (`job_skills.json`) for reference during analysis.

---

## Project Structure

Below is a sample layout of the project's folder structure:

	└──Resume_Analyser_for_Job_Description-2024-2025-Project/
	│── Team 1/
	│   ├── app.py
	│   ├── model.py
	│   ├── requirements.txt
	│   ├── output/
	│   ├── .streamlit/
	│   ├── README.md
	│── Team 2/
	│   ├── ResumeAnalyser.ipynb
	│   ├── trained_model.pkl
	│   ├── trained_model_with_ats.pkl
	│   ├── vectorizer.pkl
	│   ├── index.html
	│   ├── README.md
	│── main_README.md




## Getting Started

### Prerequisites

- **Programming Language:** Python 3.7 or higher
- **Package Manager:** Pip
- **Required Libraries:** Streamlit, pandas, scikit-learn, spaCy, pymupdf, PyPDF2, language_tool_python, textblob, and others listed in [requirements.txt](requirements.txt).

### Installation

1. **Clone the Repository:**

       git clone https://github.com/yourusername/Resume_Analyser_for_Job_Description-2024-2025-Project.git

2. **Navigate to the Project Directory:**
   
       cd Team 1
       cd Resume_Analyser_for_Job_Description-2024-2025-Project
       

4. **Install Dependencies:**

       pip install -r requirements.txt
       python -m spacy download en_core_web_sm

### Usage

To launch the Resume Ranking Application, use the following command:

    streamlit run app.py

This will start a local Streamlit server where you can:
- Upload a ZIP file containing PDF resumes.
- Optionally upload a job description (TXT file).
- View the analysis results, download the final ranked CSV, and retrieve the top ranked resume PDFs.

---

## Project Roadmap

- [X] **Task 1:** Implement resume extraction and text conversion from PDF files.
- [X] **Task 2:** Enhance scoring algorithms with additional metrics and dynamic weighting.
- [X] **Task 3:** Integrate more robust job description parsing and candidate matching features.
- [X] **Task 4:** Improve UI/UX and add more visualization options in the Streamlit app.

---

## Contributing

- **💬 [Join the Discussions](https://LOCAL/GitHub/Resume_Analyser_for_Job_Description-2024-2025-Project/discussions):** Share insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/GitHub/Resume_Analyser_for_Job_Description-2024-2025-Project/issues):** Submit bugs or request features.
- **💡 [Submit Pull Requests](https://LOCAL/GitHub/Resume_Analyser_for_Job_Description-2024-2025-Project/blob/main/CONTRIBUTING.md):** Fork the repository, create a feature branch, and submit a PR.

<details>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository:** Fork the project to your account.  
2. **Clone Locally:** Clone your forked repository.  
       git clone https://github.com/yourusername/Resume_Analyser_for_Job_Description-2024-2025-Project.git 
3. **Create a New Branch:**  
       git checkout -b new-feature-x  
4. **Make Your Changes:** Develop and test your changes locally.  
5. **Commit Your Changes:**  
       git commit -m "Implemented feature x."  
6. **Push to Your Fork:**  
       git push origin new-feature-x  
7. **Submit a Pull Request:** Create a PR against the original repository with a clear description of your changes.  
8. **Review:** Once reviewed and approved, your changes will be merged.  

</details>

---

## Future Plans

- Improve the regex and pattern matching for extraction
- Enhance the NLP pipeline
- Host on AWS/Google Cloud
- Suggest improvements to resumes
- Generate cover letters
- Incorporate weightage to personal recommendations on the basis of content of recommendation letters

---

## License

This project is protected under the [MIT](https://choosealicense.com/licenses/mit/#) License. For more details, please refer to the [LICENSE](https://choosealicense.com/licenses/) site.

---


## Acknowledgments

- **Contributors:**
  - **Samudraneel Sarkar**  
    [LinkedIn](https://www.linkedin.com/in/samudraneel-sarkar) | [GitHub](https://github.com/samudraneel05) 
  - **Guransh Goyal**  
    [LinkedIn](https://www.linkedin.com/in/guransh-goyal) | [GitHub](https://github.com/GuranshGoyal) 
  - **Yash Kesarwani**  
    [LinkedIn](https://www.linkedin.com/in/yash-keswani-856b45293) | [GitHub](https://github.com/yashkeswani29-cpu) 

- **Inspiration & Contributions:** Thanks to the open-source community for providing robust libraries (such as spaCy, scikit-learn, and Streamlit) that made this project possible.
- **Other Resources:** Special thanks to language processing libraries like [language_tool_python](https://pypi.org/project/language-tool-python/) and [TextBlob](https://textblob.readthedocs.io/en/dev/) which help enhance the resume quality evaluation.

    
