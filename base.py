from gerenciador import Gerenciador
from vetorizador import CountVectorizer, TFIDFVectorizer
from classificador import NaiveBayes, SVM
from util import analisar_sentimento

gerenciador = Gerenciador()
revisoes = gerenciador.revisoes
recomendacoes = gerenciador.recomendacoes

vec1 = CountVectorizer(revisoes)
vec2 = CountVectorizer(revisoes, (1,4))
vec3 = TFIDFVectorizer(revisoes)
vec4 = TFIDFVectorizer(revisoes, (1,4))

class1 = NaiveBayes("naive_bayes_1_1.pickle", vec1, recomendacoes)
class2 = NaiveBayes("naive_bayes_1_4.pickle", vec2, recomendacoes)
class3 = SVM("svm_1_1.pickle", vec3, recomendacoes)
class4 = SVM("svm_1_4.pickle", vec4, recomendacoes)

testar = True

while testar is True:
    texto = input("\n>>> ")

    if texto == 0:
        testar = False
    else:
        analisar_sentimento(class1, texto)
        analisar_sentimento(class2, texto)
        analisar_sentimento(class3, texto)
        analisar_sentimento(class4, texto)


#class1 = NaiveBayes("naive_bayes_1_1.pickle")
#class2 = NaiveBayes("naive_bayes_1_4.pickle")
#class3 = SVM("svm_1_1.pickle")
#class4 = SVM("svm_1_4.pickle")

#print(class1.marcador())
#print(class2.marcador())
#print(class3.marcador())
#print(class4.marcador())
