# OneAPI hackathon for creating QA chatbot

##

A python application which contains various python libraries which will generate answers based on the question.

## Features of the python libraries

- pandas: Used for reading and writing CSV files.
- transformers: Used for importing and loading LLMs.
- tqdm: Used for generating application stats bar.
- torch: Used to run the models along with transformers library.
- streamlit: Used for creating an interative application.

## Tech

- [pandas](https://pypi.org)
- [transformers](https://pypi.org)
- [tqdm](https://pypi.org)
- [torch](https://pypi.org)
- [streamlit](https://pypi.org)

## Installation

Dependencies

- python: "3.11.x",
- pandas: "2.1.2",
- transformers: "4.34.1",
- tqdm: "4.66.1",
- torch: "2.1.0",
- streamlit: "1.28.0"
  Install the dependencies and devDependencies and start the server.

For dev environments...

```sh
pip install -r requirements.txt
streamlit run streamlit.py
```

## Testing

## How to Run the oneAPI

To run this oneAPI on your local machine, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have python 3.11 and pip installed.
3. Install oneAPI dependencies by running `pip install -r requirements.txt` in the oneAPI directory.
4. Start the server using `streamlit run streamlit.py`.

The server will start and listen on the specified port. You can access the microservices using the provided endpoints as described below. Ensure you replace placeholders with actual values in your environment variables, such as database connection details and API keys.

## API Usage

You can use browsers like google or edge to interact with the web application. Here are the details for web application:

- http://localhost:8501
