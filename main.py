from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM




load_dotenv()



def main():
    print("Hello from langchain-course!")
    
    # region Stephen Hawking Information
    information = """Stephen William Hawking (pronunciación en inglés: /stiːvən_ˈhɔːkɪŋ/ (escucharⓘ); Oxford, 8 de enero de 1942-Cambridge, 14 de marzo de 2018)[2]​[3]​ fue un físico teórico, astrofísico, cosmólogo y divulgador científico británico. Sus trabajos más importantes consistieron en aportar, junto con Roger Penrose, teoremas respecto a las singularidades espaciotemporales en el marco de la relatividad general y la predicción teórica de que los agujeros negros emitirían radiación,[4]​ lo que se conoce hoy en día como radiación de Hawking (o a veces radiación Bekenstein-Hawking). Una de las principales características de su personalidad fue su contribución al debate científico, a veces apostando públicamente con otros científicos. El caso más conocido es su participación en la discusión sobre la conservación de la información en los agujeros negros.[5]​

    Era miembro de la Real Sociedad de Londres, de la Pontificia Academia de las Ciencias y de la Academia Nacional de Ciencias (Estados Unidos). Fue titular de la Cátedra Lucasiana de Matemáticas (Lucasian Chair of Mathematics) de la Universidad de Cambridge desde 1979 hasta su jubilación en 2009.[6]​

    Entre las numerosas distinciones que le fueron concedidas, recibió doce doctorados honoris causa y fue galardonado con la Orden del Imperio Británico (grado CBE) en 1982, el Premio Príncipe de Asturias de la Concordia en 1989, la Medalla Copley en 2006, la Medalla de la Libertad en 2009[7]​ y el Premio Fundación BBVA Fronteras del Conocimiento en 2015.

    Estuvo casado en dos ocasiones y tuvo tres hijos. Justo antes de su primer matrimonio, con veintiún años, se le diagnósticó esclerosis lateral amiotrófica (ELA), que fue agravando su estado con el paso de los años, hasta dejarlo casi completamente paralizado[8]​[4]​ y le forzó a comunicarse a través de un aparato generador de voz. Ha sido la persona de más edad con esta enfermedad, a la que sobrevivió cincuenta y cinco años, cuando la esperanza media de vida es de aproximadamente catorce meses.[8]​[9]​ Su caso resulta «fascinante» y desconcertante para los neurólogos.[8]​

    Como autor de libros divulgativos sobre ciencia, alcanzó enormes éxitos de ventas, en los que discute sobre sus propias teorías y la cosmología en general, como Breve historia del tiempo: del Big Bang a los agujeros negros (A Brief History of Time: From the Big Bang to Black Holes), de 1988, y que estuvo en la lista de superventas del The Sunday Times británico durante 237 semanas; Brevísima historia del tiempo (A Briefer History of Time), de 2005, en colaboración con Leonard Mlodinow, en la que trató de explicar de la manera más sencilla posible la historia del universo, motivo por el cual se le conoció como El historiador del tiempo[10]​[11]​ o El historiador del universo,[12]​ y El universo en una cáscara de nuez (The Universe in a Nutshell), de 2001."""
    # endregion
    # region summary_template
    summary_template = """
    Given the information {information} about a person, i want you to create:
    1. summary of the person
    2. two interesting facts about the person
    """
    # endregion
    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"])
    llm_openai = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
    llm_ollama = OllamaLLM(temperature=0.5, model="gemma3:270m")
    chain = summary_prompt_template | llm_ollama # chain = LLMChain(prompt=summary_prompt_template, llm=llm)
    
    summary = chain.invoke({"information": information})
    print(summary)

if __name__ == "__main__":
    main()
