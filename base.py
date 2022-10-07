from gerenciador import Gerenciador
from vetorizador import CountVectorizer, TFIDFVectorizer

gerenciador = Gerenciador()
revisoes = gerenciador.revisoes
recomendacoes = gerenciador.recomendacoes

vec1 = CountVectorizer(revisoes)
vec2 = CountVectorizer(revisoes, (1,4))
vec3 = TFIDFVectorizer(revisoes)
vec4 = TFIDFVectorizer(revisoes, (1,4))

print(len(vec1.vetorizador.vocabulary_))
print(len(vec2.vetorizador.vocabulary_))
print(len(vec3.vetorizador.vocabulary_))
print(len(vec4.vetorizador.vocabulary_))

