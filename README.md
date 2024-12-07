```python
from IPython.display import Image
```


```python
Image(filename = 'Simple-Agent.jpeg')
```




    
![jpeg](README_files/README_1_0.jpg)
    



# Simple Agent

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: [Apache-2.0](https://github.com/mytechnotalent/Simple-Agent/blob/main/LICENSE)

## Install Libraries


```python
!pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface
```

    Requirement already satisfied: llama-index-core in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.12.2)
    Requirement already satisfied: llama-index-readers-file in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.4.1)
    Requirement already satisfied: llama-index-llms-ollama in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.4.2)
    Requirement already satisfied: llama-index-embeddings-huggingface in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (0.4.0)
    Requirement already satisfied: PyYAML>=6.0.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (6.0.1)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core) (2.0.34)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (3.10.5)
    Requirement already satisfied: dataclasses-json in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (0.6.7)
    Requirement already satisfied: deprecated>=1.2.9.3 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.2.15)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.0.8)
    Requirement already satisfied: filetype<2.0.0,>=1.2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.2.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (2024.6.1)
    Requirement already satisfied: httpx in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (0.27.0)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (3.3)
    Requirement already satisfied: nltk>3.8.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (3.9.1)
    Requirement already satisfied: numpy in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.26.4)
    Requirement already satisfied: pillow>=9.0.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (10.4.0)
    Requirement already satisfied: pydantic<2.10.0,>=2.7.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (2.9.2)
    Requirement already satisfied: requests>=2.31.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (2.32.3)
    Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (8.2.3)
    Requirement already satisfied: tiktoken>=0.3.3 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (0.8.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (4.66.5)
    Requirement already satisfied: typing-extensions>=4.5.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (4.12.2)
    Requirement already satisfied: typing-inspect>=0.8.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (0.9.0)
    Requirement already satisfied: wrapt in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-core) (1.14.1)
    Requirement already satisfied: beautifulsoup4<5.0.0,>=4.12.3 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-readers-file) (4.12.3)
    Requirement already satisfied: pandas in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-readers-file) (2.2.2)
    Requirement already satisfied: pypdf<6.0.0,>=5.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-readers-file) (5.1.0)
    Requirement already satisfied: striprtf<0.0.27,>=0.0.26 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-readers-file) (0.0.26)
    Requirement already satisfied: ollama>=0.3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-llms-ollama) (0.4.2)
    Requirement already satisfied: huggingface-hub>=0.19.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (0.26.2)
    Requirement already satisfied: sentence-transformers>=2.6.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from llama-index-embeddings-huggingface) (3.3.0)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (2.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (1.2.0)
    Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (23.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (1.4.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core) (1.11.0)
    Requirement already satisfied: soupsieve>1.2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from beautifulsoup4<5.0.0,>=4.12.3->llama-index-readers-file) (2.5)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.13.1)
    Requirement already satisfied: packaging>=20.9 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (24.1)
    Requirement already satisfied: click in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from nltk>3.8.1->llama-index-core) (8.1.7)
    Requirement already satisfied: joblib in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from nltk>3.8.1->llama-index-core) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from nltk>3.8.1->llama-index-core) (2024.9.11)
    Requirement already satisfied: anyio in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpx->llama-index-core) (4.2.0)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpx->llama-index-core) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpx->llama-index-core) (1.0.2)
    Requirement already satisfied: idna in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpx->llama-index-core) (3.7)
    Requirement already satisfied: sniffio in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpx->llama-index-core) (1.3.0)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from httpcore==1.*->httpx->llama-index-core) (0.14.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pydantic<2.10.0,>=2.7.0->llama-index-core) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pydantic<2.10.0,>=2.7.0->llama-index-core) (2.23.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests>=2.31.0->llama-index-core) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from requests>=2.31.0->llama-index-core) (2.2.3)
    Requirement already satisfied: transformers<5.0.0,>=4.41.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (4.46.2)
    Requirement already satisfied: torch>=1.11.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (2.5.1)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (1.5.1)
    Requirement already satisfied: scipy in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (1.13.1)
    Requirement already satisfied: greenlet!=0.4.17 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core) (3.0.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from typing-inspect>=0.8.0->llama-index-core) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from dataclasses-json->llama-index-core) (3.23.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas->llama-index-readers-file) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas->llama-index-readers-file) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from pandas->llama-index-readers-file) (2023.3)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-readers-file) (1.16.0)
    Requirement already satisfied: jinja2 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (3.1.4)
    Requirement already satisfied: setuptools in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (75.1.0)
    Requirement already satisfied: sympy==1.13.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from sympy==1.13.1->torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (1.3.0)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (0.20.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from scikit-learn->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (3.5.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/prod/lib/python3.12/site-packages (from jinja2->torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface) (2.1.3)


## Install & Run Ollama


```python
import os
import platform
import subprocess


def install_and_manage_ollama():
    """Install and manage Ollama on various OS plaforms."""

    # detect system
    system = platform.system()

    # detect, install and run Ollama on respective OS platform
    try:
        if system == "Darwin":
            print("Detected macOS. Checking if Ollama is installed...")
            if subprocess.run(['which', 'ollama'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
                print("Installing Ollama on macOS using Homebrew...")
                os.system("brew install ollama")
            else:
                print("Ollama is already installed.")
        elif system == "Linux":
            print("Detected Linux. Checking if Ollama is installed...")
            if subprocess.run(['which', 'ollama'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
                print("Installing Ollama on Linux...")
                os.system("curl -sSL https://ollama.com/install | sh")
            else:
                print("Ollama is already installed.")
        elif system == "Windows":
            print("Detected Windows.")
            print("Please download and install Ollama manually from https://ollama.com.")
            return
        else:
            print("Unsupported operating system. Exiting.")
            return
        print("Managing Ollama process...")
        if system in ["Darwin", "Linux"]:
            result = subprocess.run(['pgrep', '-f', 'ollama'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stdout:
                pid = result.stdout.strip()
                print(f"Found running Ollama process with PID: {pid}. Killing it...")
                os.system(f"kill -9 {pid}")
            else:
                print("No Ollama process found running.")
            print("Starting ollama serve in the background...")
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
            print("ollama serve is now running in the background.")
        else:
            print("Automatic management of Ollama is not supported on Windows. Please run Ollama manually.")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


# run the function
install_and_manage_ollama()
```

    Detected macOS. Checking if Ollama is installed...
    Ollama is already installed.
    Managing Ollama process...
    Found running Ollama process with PID: 84768
    86693. Killing it...
    Starting ollama serve in the background...
    ollama serve is now running in the background.


    sh: line 1: 86693: command not found


## Obtain `mixtral:8x7b` Model


```python
import platform
import subprocess


def run_ollama_mixtral():
    """Obtain mixtral:8x7b model from Ollama."""

    # detect system
    system = platform.system()

    # detect and obtain mixtral:8x7b model on respective OS platform
    try:
        if system in ["Darwin", "Linux"]:  
            print(f"Detected {system}. Running ollama mixtral:8x7b...")
            if subprocess.run(['which', 'ollama'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
                print("Ollama is not installed. Please install it and try again.")
                return
            result = subprocess.run(['ollama', 'run', 'mixtral:8x7b'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print("Command executed successfully.")
            else:
                print("Failed to execute the command:")
                print(result.stderr)
        elif system == "Windows":
            print("Detected Windows.")
            print("Please run the following command manually in your terminal:")
            print("`ollama run mixtral:8x7b`")
            return
        else:
            print("Unsupported operating system. Exiting.")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


# run the function
run_ollama_mixtral()
```

    Detected Darwin. Running ollama mixtral:8x7b...
    Command executed successfully.


## Register Tools w/ `ReactAgent`


```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool


def factorial(n: int) -> int:
    """Calculate the factorial of a number."""
    if n == 0:
        return 1
    return n * factorial(n - 1)


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


factorial_tool = FunctionTool.from_defaults(fn=factorial)
is_prime_tool = FunctionTool.from_defaults(fn=is_prime)

llm = Ollama(model="mixtral:8x7b", request_timeout=300.0)
agent = ReActAgent.from_tools(
    [factorial_tool, is_prime_tool], 
    llm=llm, 
    verbose=True, 
    max_iterations=10  # increase from the default (usually 3-5)
)
```

## Chat w/ Agent


```python
response = agent.chat("What is the factorial of 5? Calculate step by step detailing all of your thought process.")
print(response)
```

    > Running step 4d8f9068-906a-4486-afcf-e00f122c5a84. Step input: What is the factorial of 5? Calculate step by step detailing all of your thought process.
    [1;3;38;5;200mThought: The current language of the user is English. They want to know the factorial of 5 and also want to see a step-by-step calculation. I can use the 'factorial' tool to find the factorial of 5, but I also need to manually calculate it for demonstration purposes.
    Action: No
    Action Input: {'n': 5}
    [0m[1;3;34mObservation: Error: No such tool named `No`.
    [0m> Running step 73e48c92-2ed0-4d26-a27c-4506b658f9b4. Step input: None
    [1;3;38;5;200mThought: The user received an error because there is no "No" tool. I should have written "No tool" instead. I will clarify this in my next message and then show the step-by-step calculation of the factorial of 5.
    Answer: I'm sorry for the confusion. I made a mistake in my previous response by mentioning "No" as the action. To answer your question, let's manually calculate the factorial of 5 (denoted as 5!). The factorial of a number is calculated by multiplying it by all positive integers below it. So,
    
    1 × 2 × 3 × 4 × 5 = 120
    
    Therefore, the factorial of 5 is 120. I will now use the 'factorial' tool to confirm this result.
    Action: factorial
    Action Input: {'n': 5}
    [0m[1;3;34mObservation: 120
    [0m> Running step 8f1d6e30-eada-4b8c-afb3-2e2693031f1b. Step input: None
    [1;3;38;5;200mThought: The 'factorial' tool returned the correct value, 120, which confirms our manual calculation of the factorial of 5. To summarize, the factorial of 5 is indeed 120.
    Answer: I can confirm that the factorial of 5 is 120. I used both a manual calculation and the 'factorial' tool to determine this result. Here's how we calculated it step-by-step:
    
    1 × 2 × 3 × 4 × 5 = 120
    
    This confirms that the factorial of 5 is 120. If you have any other questions, feel free to ask!
    [0mI can confirm that the factorial of 5 is 120. I used both a manual calculation and the 'factorial' tool to determine this result. Here's how we calculated it step-by-step:
    
    1 × 2 × 3 × 4 × 5 = 120
    
    This confirms that the factorial of 5 is 120. If you have any other questions, feel free to ask!



```python
response = agent.chat("Is 29 a prime number? Calculate step by step detailing all of your thought process.")
print(response)
```

    > Running step 8295ef74-8fea-4936-ae7a-17eb52aecf5e. Step input: Is 29 a prime number? Calculate step by step detailing all of your thought process.
    [1;3;38;5;200mThought: (Implicit) I can answer without any more tools!
    Answer: To determine if a number is prime, we need to check if it has any divisors other than 1 and itself. Here's the step-by-step calculation for checking if 29 is a prime number:
    
    1. Start with the number 2 (the smallest possible prime factor).
    2. Check if 2 is a divisor of 29 (i.e., 29 % 2 == 0). If it is, then 29 is not a prime number because it has a divisor other than 1 and itself. However, in this case, 29 % 2 != 0, which means that 2 is not a divisor of 29.
    3. Move on to the next possible prime factor, which is 3. Check if 3 is a divisor of 29 (i.e., 29 % 3 == 0). Again, in this case, 29 % 3 != 0, so 3 is not a divisor of 29.
    4. Continue checking for higher prime factors up to the square root of 29 (approximately 5.39). Since none of them are divisors of 29, we can conclude that 29 is indeed a prime number.
    
    So, 29 is a prime number. I arrived at this conclusion by manually performing the steps to check for any divisors other than 1 and itself.
    [0mTo determine if a number is prime, we need to check if it has any divisors other than 1 and itself. Here's the step-by-step calculation for checking if 29 is a prime number:
    
    1. Start with the number 2 (the smallest possible prime factor).
    2. Check if 2 is a divisor of 29 (i.e., 29 % 2 == 0). If it is, then 29 is not a prime number because it has a divisor other than 1 and itself. However, in this case, 29 % 2 != 0, which means that 2 is not a divisor of 29.
    3. Move on to the next possible prime factor, which is 3. Check if 3 is a divisor of 29 (i.e., 29 % 3 == 0). Again, in this case, 29 % 3 != 0, so 3 is not a divisor of 29.
    4. Continue checking for higher prime factors up to the square root of 29 (approximately 5.39). Since none of them are divisors of 29, we can conclude that 29 is indeed a prime number.
    
    So, 29 is a prime number. I arrived at this conclusion by manually performing the steps to check for any divisors other than 1 and itself.

