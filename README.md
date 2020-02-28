# nlp-quality-estimiation

Natural Language Processing (NLP hereafter) is taken the world by storm, it becomes the magic thing that makes people lives easier. One of the famous applications of NLP is Machine Translation, which aims at accurately expressing sentences in different languages. The input to the Machine Translation models is a sentence, and the output is, as well, a sentence but in different language. The main challenge with training Machine Translation models is finding the optimal output, and evaluating the predicted one. This usually involve lots of human effort which is not always possible. The field of Machine Translation Quality Estimation is becoming a very active area of research, which aims at building a Machine Learning (ML) based model that predicts the quality of the translated sentence. 

In this project, we implemented different approaches to tackle this problem, starting from the use of traditional ML approaches, and concluding with more sophisticated ones that perform better in the provided data-set. 


To run the project, please follow the following steps:

1. Clone the repo by using the following command:

    `git clone https://github.com/raghada/nlp-quality-estimiation.git`

2. Create a new python enviroment:

    `python3 -m venv nlp-env`

3. Activate the enviroment:

    `source nlp-env/bin/activate`

4. Install requirmenets.txt, using the following command:

    `pip install -r requirements.txt`

5. Run the project with this command:

    `python3 main.py --strategy 4`

Note: sending the value 4 will run the project with all strategies, if you wish to specifiy a strategy, please indicate its number (1, 2, or 3)


