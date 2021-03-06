import streamlit as st

st.set_page_config(layout="wide")

st.header("Contract Understanding Atticus Dataset (CUAD) Demo")
st.write("This demo uses a machine learning model for Contract Understanding.")

add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Hello, Welcome!")



### Import Dependencies
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

### Select the language model
transformer_name = "bert-large-nli-stsb-mean-tokens" #@param ['universal-sentence-encoder', 'roberta-large-nli-stsb-mean-tokens','roberta-large-nli-mean-tokens', 'roberta-base-nli-wkpooling', 'roberta-base-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-wkpooling', 'distilbert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens', 'bert-base-nli-stsb-wkpooling', 'bert-base-nli-stsb-mean-tokens']
if transformer_name == 'universal-sentence-encoder':
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
else:
    embed = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

### Enter the first set of paragraphs:
paragraph_1 = "In all judgements wherein the relation of a subject to the predicate is cogitated (I mention affirmative judgements only here; the application to negative will be very easy), this relation is possible in two different ways. Either the predicate B belongs to the subject A, as somewhat which is contained (though covertly) in the conception A; or the predicate B lies completely out of the conception A, although it stands in connection with it. In the first instance, I term the judgement analytical, in the second, synthetical. Analytical judgements (affirmative) are therefore those in which the connection of the predicate with the subject is cogitated through identity; those in which this connection is cogitated without identity, are called synthetical judgements. The former may be called explicative, the latter augmentative judgements; because the former add in the predicate nothing to the conception of the subject, but only analyse it into its constituent conceptions, which were thought already in the subject, although in a confused manner; the latter add to our conceptions of the subject a predicate which was not contained in it, and which no analysis could ever have discovered therein. For example, when I say, \u201CAll bodies are extended,\u201D this is an analytical judgement. For I need not go beyond the conception of body in order to find extension connected with it, but merely analyse the conception, that is, become conscious of the manifold properties which I think in that conception, in order to discover this predicate in it: it is therefore an analytical judgement. On the other hand, when I say, \u201CAll bodies are heavy,\u201D the predicate is something totally different from that which I think in the mere conception of a body. By the addition of such a predicate, therefore, it becomes a synthetical judgement." #@param {type:"string"}
paragraph_2 = "Judgements of experience, as such, are always synthetical. For it would be absurd to think of grounding an analytical judgement on experience, because in forming such a judgement I need not go out of the sphere of my conceptions, and therefore recourse to the testimony of experience is quite unnecessary. That \u201Cbodies are extended\u201D is not an empirical judgement, but a proposition which stands firm \xE0 priori. For before addressing myself to experience, I already have in my conception all the requisite conditions for the judgement, and I have only to extract the predicate from the conception, according to the principle of contradiction, and thereby at the same time become conscious of the necessity of the judgement, a necessity which I could never learn from experience. On the other hand, though at first I do not at all include the predicate of weight in my conception of body in general, that conception still indicates an object of experience, a part of the totality of experience, to which I can still add other parts; and this I do when I recognize by observation that bodies are heavy. I can cognize beforehand by analysis the conception of body through the characteristics of extension, impenetrability, shape, etc., all which are cogitated in this conception. But now I extend my knowledge, and looking back on experience from which I had derived this conception of body, I find weight at all times connected with the above characteristics, and therefore I synthetically add to my conceptions this as a predicate, and say, \u201CAll bodies are heavy.\u201D Thus it is experience upon which rests the possibility of the synthesis of the predicate of weight with the conception of body, because both conceptions, although the one is not contained in the other, still belong to one another (only contingently, however), as parts of a whole, namely, of experience, which is itself a synthesis of intuitions." #@param {type:"string"}
paragraph_3 = "But to synthetical judgements \xE0 priori, such aid is entirely wanting. If I go out of and beyond the conception A, in order to recognize another B as connected with it, what foundation have I to rest on, whereby to render the synthesis possible? I have here no longer the advantage of looking out in the sphere of experience for what I want. Let us take, for example, the proposition, \u201CEverything that happens has a cause.\u201D In the conception of \u201Csomething that happens,\u201D I indeed think an existence which a certain time antecedes, and from this I can derive analytical judgements. But the conception of a cause lies quite out of the above conception, and indicates something entirely different from \u201Cthat which happens,\u201D and is consequently not contained in that conception. How then am I able to assert concerning the general conception\u2014\u201Cthat which happens\u201D\u2014something entirely different from that conception, and to recognize the conception of cause although not contained in it, yet as belonging to it, and even necessarily? what is here the unknown = X, upon which the understanding rests when it believes it has found, out of the conception A a foreign predicate B, which it nevertheless considers to be connected with it? It cannot be experience, because the principle adduced annexes the two representations, cause and effect, to the representation existence, not only with universality, which experience cannot give, but also with the expression of necessity, therefore completely \xE0 priori and from pure conceptions. Upon such synthetical, that is augmentative propositions, depends the whole aim of our speculative knowledge \xE0 priori; for although analytical judgements are indeed highly important and necessary, they are so, only to arrive at that clearness of conceptions which is requisite for a sure and extended synthesis, and this alone is a real acquisition." #@param {type:"string"}
paragraph_4 = 'Mathematical judgements are always synthetical. Hitherto this fact, though incontestably true and very important in its consequences, seems to have escaped the analysts of the human mind, nay, to be in complete opposition to all their conjectures. For as it was found that mathematical conclusions all proceed according to the principle of contradiction (which the nature of every apodeictic certainty requires), people became persuaded that the fundamental principles of the science also were recognized and admitted in the same way. But the notion is fallacious; for although a synthetical proposition can certainly be discerned by means of the principle of contradiction, this is possible only when another synthetical proposition precedes, from which the latter is deduced, but never of itself.' #@param {type:"string"}
paragraph_5 = "Before all, be it observed, that proper mathematical propositions are always judgements \xE0 priori, and not empirical, because they carry along with them the conception of necessity, which cannot be given by experience. If this be demurred to, it matters not; I will then limit my assertion to pure mathematics, the very conception of which implies that it consists of knowledge altogether non-empirical and \xE0 priori." #@param {type:"string"}
paragraph_6 = "We might, indeed at first suppose that the proposition 7 + 5 = 12 is a merely analytical proposition, following (according to the principle of contradiction) from the conception of a sum of seven and five. But if we regard it more narrowly, we find that our conception of the sum of seven and five contains nothing more than the uniting of both sums into one, whereby it cannot at all be cogitated what this single number is which embraces both. The conception of twelve is by no means obtained by merely cogitating the union of seven and five; and we may analyse our conception of such a possible sum as long as we will, still we shall never discover in it the notion of twelve. We must go beyond these conceptions, and have recourse to an intuition which corresponds to one of the two\u2014our five fingers, for example, or like Segner in his Arithmetic five points, and so by degrees, add the units contained in the five given in the intuition, to the conception of seven. For I first take the number 7, and, for the conception of 5 calling in the aid of the fingers of my hand as objects of intuition, I add the units, which I before took together to make up the number 5, gradually now by means of the material image my hand, to the number 7, and by this process, I at length see the number 12 arise. That 7 should be added to 5, I have certainly cogitated in my conception of a sum = 7 + 5, but not that this sum was equal to 12. Arithmetical propositions are therefore always synthetical, of which we may become more clearly convinced by trying large numbers. For it will thus become quite evident that, turn and twist our conceptions as we may, it is impossible, without having recourse to intuition, to arrive at the sum total or product by means of the mere analysis of our conceptions. Just as little is any principle of pure geometry analytical. \u201CA straight line between two points is the shortest,\u201D is a synthetical proposition. For my conception of straight contains no notion of quantity, but is merely qualitative. The conception of the shortest is therefore fore wholly an addition, and by no analysis can it be extracted from our conception of a straight line. Intuition must therefore here lend its aid, by means of which, and thus only, our synthesis is possible." #@param {type:"string"}
paragraph_7 = "Some few principles preposited by geometricians are, indeed, really analytical, and depend on the principle of contradiction. They serve, however, like identical propositions, as links in the chain of method, not as principles\u2014for example, a = a, the whole is equal to itself, or (a+b) \u2014> a, the whole is greater than its part. And yet even these principles themselves, though they derive their validity from pure conceptions, are only admitted in mathematics because they can be presented in intuition. What causes us here commonly to believe that the predicate of such apodeictic judgements is already contained in our conception, and that the judgement is therefore analytical, is merely the equivocal nature of the expression. We must join in thought a certain predicate to a given conception, and this necessity cleaves already to the conception. But the question is, not what we must join in thought to the given conception, but what we really think therein, though only obscurely, and then it becomes manifest that the predicate pertains to these conceptions, necessarily indeed, yet not as thought in the conception itself, but by virtue of an intuition, which must be added to the conception." #@param {type:"string"}
paragraph_8 = "The science of natural philosophy (physics) contains in itself synthetical judgements \xE0 priori, as principles. I shall adduce two propositions. For instance, the proposition, \u201CIn all changes of the material world, the quantity of matter remains unchanged\u201D; or, that, \u201CIn all communication of motion, action and reaction must always be equal.\u201D In both of these, not only is the necessity, and therefore their origin \xE0 priori clear, but also that they are synthetical propositions. For in the conception of matter, I do not cogitate its permanency, but merely its presence in space, which it fills. I therefore really go out of and beyond the conception of matter, in order to think on to it something \xE0 priori, which I did not think in it. The proposition is therefore not analytical, but synthetical, and nevertheless conceived \xE0 priori; and so it is with regard to the other propositions of the pure part of natural philosophy." #@param {type:"string"}
paragraphs1 = [paragraph_1, paragraph_2, paragraph_3, paragraph_4, paragraph_5, paragraph_6, paragraph_7, paragraph_8]

### Enter the second set of paragraphs:
paragraph_1 = "OLD ZELIG was eyed askance by his brethren. No one deigned to call him \"Reb\" Zelig, nor to prefix to his name the American equivalent \u2014 \"Mr.\" \"The old one is a barrel with a stave missing,\" knowingly declared his neighbors. \"He never spends a cent; and he belongs nowheres.\" For \"to belong,\" on New York's East Side, is of no slight importance. It means being a member in one of the numberless congregations. Every decent Jew must join \"A Society for Burying Its Members,\" to be pro- vided at least with a narrow cell at the end of the long road. Zelig was not even a member of one of these. \"Alone, like a stone,\" his wife often sighed." #@param {type:"string"}
paragraph_2 = "In the cloakshop where Zelig worked he stood daily, brandishing his heavy iron on the sizzling cloth, hardly ever glancing about him. The workmen despised him, for during a strike he returned to work after two days' absence. He could not be idle, and thought with dread of the Saturday that would bring him no pay envelope." #@param {type:"string"}
paragraph_3 = "His very appearance seemed alien to his brethren. His figure was tall, and of cast-iron mold. When he stared stupidly at something, he looked like a blind Samson. His gray hair was long, and it fell in disheveled curls on gigantic shoulders somewhat inclined to stoop. His shabby clothes hung loosely on him; and, both summer and winter, the same old cap covered his massive head." #@param {type:"string"}
paragraph_4 = "He had spent most of his life in a sequestered village in Little Russia, where he tilled the soil and even wore the national peasant costume. When his son and only child, a poor widower with a boy of twelve on his hands, emigrated to America, the father's heart bled. Yet he chose to stay in his native village at all hazards, and to die there. One day, how- ever, a letter arrived from the son that he was sick; this sad news was followed by words of a more cheerful nature \u2014 \"and your grandson Moses goes to public school. He is almost an American; and he is not forced to forget the God of Israel. He will soon be confirmed. His Bar Mitsva is near.\" Zelig's wife wept three days and nights upon the receipt of this letter. The old man said little; but he began to sell his few possessions." #@param {type:"string"}
paragraph_5 = "To face the world outside his village spelled agony to the poor rustic. Still he thought he would get used to the new home which his son had chosen. But the strange journey with locomotive and steamship bewildered him dreadfully; and the clamor of the metropolis, into which he was flung pell-mell, altogether stupefied him. With a vacant air he regarded the Pandemonium, and a petrifaction of his inner being seemed to take place. He became \"a barrel with a stave missing.\" No spark of animation visited his eye. Only one thought survived in his brain, and one desire pulsed in his heart: to save money enough for himself and family to hurry back to his native village. Blind and dead to everything, he moved about with a dumb, lacerating pain in his heart, \u2014 he longed for home. Before he found steady employment, he walked daily with titanic strides through the entire length of Manhattan, while children and even adults often slunk into byways to let him pass. Like a huge monster he seemed, with an arrow in his vitals." #@param {type:"string"}
paragraph_6 = "In the shop where he found a job at last, the workmen feared him at first; but, ultimately finding him a harmless giant, they more than once hurled their sarcasms at his head. Of the many men and women em- ployed there, only one person had the distinction of getting fellowship from old Zelig. That person was the Gentile watchman or janitor of the shop, a little blond Pole with an open mouth and frightened eyes. And many were the witticisms aimed at this uncouth pair. \"The big one looks like an elephant,\" the joker of the shop would say; \"only he likes to be fed on pennies instead of peanuts.\"" #@param {type:"string"}
paragraph_7 = "\"Oi, oi, his nose would betray him,\" the \"philosopher\" of the shop chimed in; and during the dinner hour he would expatiate thus: \"You see, money is his blood. He starves himself to have enough dollars to go back to his home; the Pole told me all about it. And why should he stay here? Freedom of religion means nothing to him, he never goes to syna- gogue; and freedom of the press? Bah \u2014 he never even reads the con- servative Tageblatt!\"" #@param {type:"string"}
paragraph_8 = "Old Zelig met such gibes with stoicism. Only rarely would he turn up the whites of his eyes, as if in the act of ejaculation; but he would soon contract his heavy brows into a scowl and emphasize the last with a heavy thump of his sizzling iron." #@param {type:"string"}
paragraphs2 = [paragraph_1, paragraph_2, paragraph_3, paragraph_4, paragraph_5, paragraph_6, paragraph_7, paragraph_8]

### Enter the third set of paragraphs:
paragraph_1 = "On the day when Isaac was weaned, Abraham made a great feast, to which he invited all the people of the land. Not all of those who came to enjoy the feast believed in the alleged occasion of its celebration, for some said contemptuously, {27}\"This old couple have adopted a foundling, and provided a feast to persuade us to believe that the child is their own offspring.\" What did Abraham do? He invited all the great men of the day, and Sarah invited their wives, who brought their infants, but not their nurses, along with them. On this occasion Sarah's breasts became like two fountains, for she supplied, of her own body, nourishment to all the children. Still some were unconvinced, and said, \"Shall a child be born to one that is a hundred years old, and shall Sarah, who is ninety years old, bear?\" (Gen. xvii. 17.) Whereupon, to silence this objection, Isaac's face was changed, so that it became the very picture of Abraham's; then one and all exclaimed, \"Abraham begat Isaac.\"" #@param {type:"string"}
paragraph_2 = "Rava relates the following in the name of Rabbi Yochanan:\u2014\"Two Jewish slaves were one day walking along, when their master, who was following, overheard the one saying to the other, 'There is a camel ahead of us, as I judge\u2014for I have not seen\u2014that is blind of one eye and laden with two skin-bottles, one of which contains wine and the other oil, while two drivers attend it, one of them an Israelite, and the other a Gentile.' 'You perverse men,' said their master, 'how can you fabricate such a story as that?' The slave answered, and gave this as his reason, 'The grass is cropped only on one side of the track, the wine, that must have dripped, has soaked into the earth on the right, and the oil has trickled down, and may be seen on the left; while one of the drivers turned aside from the track to ease himself, but the other has not even left the road for the purpose.' Upon this the master stepped on before them in order to verify the correctness of their inferences, and found the conclusion true in every particular. He then turned back, and ... after complimenting the two slaves for their shrewdness, he at once gave them their liberty.\"" #@param {type:"string"}
paragraph_3 = "When an Israelite and a Gentile have a lawsuit before thee, if thou canst, acquit the former according to the laws of Israel, and tell the latter such is our law; if thou canst get him off in accordance with Gentile law, do so, and say to the plaintiff such is your law; but if he cannot be acquitted according to either law, then bring forward adroit pretexts and secure his acquittal. These are the words of the Rabbi Ishmael. Rabbi Akiva says, \"No false pretext should be brought forward, because, if found out, the name of God would be blasphemed; but if there be no fear of that, then it may be adduced.\"" #@param {type:"string"}
paragraph_4 = "The stone which Og, king of Bashan, meant to throw upon Israel is the subject of a tradition delivered on Sinai. \"The camp of Israel I see,\" he said, \"extends three miles; I shall therefore go and root up a mountain three miles in extent and throw it upon them.\" So off he went, and finding such a mountain, raised it on his head, but the Holy One\u2014blessed be He!\u2014sent an army of ants against him, which so bored the mountain over his head that it slipped down upon his shoulders, from which he could not lift it, because his teeth, protruding, had riveted it upon him. This explains that which is written (Ps. iii. 7), \"Thou hast broken the teeth of the ungodly;\" where read not \"Thou hast broken,\" but \"Thou hast ramified,\" that is, \"Thou hast caused to branch out.\" Moses being ten ells in height, seized an axe ten ells long, and springing up ten ells, struck a blow on Og's ankle and killed him." #@param {type:"string"}
paragraph_5 = "Rav Yehudah used to say, \"Three things shorten a man's days and years:\u20141. Neglecting to read the law when it is given to him for that purpose; seeing it is written (Deut. xxx. 20), 'For He (who gave it) is thy life and the length of thy days.' 2. Omitting to repeat the customary benediction over a cup of blessing; for it is written (Gen. xii. 3), 'And I will bless them that bless thee.' {35}3. And the assumption of a Rabbinical air; for Rabbi Chama bar Chanena says, 'Joseph died before any of his brethren, because he domineered over them.'\"" #@param {type:"string"}
paragraph_6 = "Three things proceed by pre-eminence from God Himself:\u2014Famine, plenty, and a wise ruler. Famine (2 Kings viii. 2): \"The Lord hath called for a famine;\" plenty (Ezek. xxxvi. 29): \"I will call for corn and increase it;\" a wise ruler; for it is written (Exod. xxxi. 2), \"I have called by name Bezaleel.\" Rabbi Yitzchak says, \"A ruler is not to be appointed unless the community be first consulted. God first consulted Moses, then Moses consulted the nation concerning the appointment of Bezaleel.\"" #@param {type:"string"}
paragraph_7 = "It were better to cut the hands off than to touch the eye, or the nose, or the mouth, or the ear, etc., with them without having first washed them. Unwashed hands may cause blindness, deafness, foulness of breath, or a polypus. {36}It is taught that Rabbi Nathan has said, \"The evil spirit Bath Chorin, which rests upon the hands at night, is very strict; he will not depart till water is poured upon the hands three times over.\"" #@param {type:"string"}
paragraph_8 = "There are three whom the Holy One\u2014blessed be He!\u2014Himself proclaims virtuous:\u2014The unmarried man who lives in a city and does not sin; the poor man who restores a lost thing which he has found to its owner; and the rich man who pays the tithes of his increase unostentatiously. Rav Saphra was a bachelor, and he dwelt in a large city. A disciple of the wise once descanted upon the merits of a celibate life in the presence of Rava and this Rav Saphra, and the face of the latter beamed with delight. Remarking which, Rava said to him, \"This does not refer to such a bachelor as thou art, but to such as Rabbi Chanena and Rabbi Oshaia.\" They were single men, who followed the trade of shoemakers, and dwelt in a street mostly occupied by meretrices, for whom they made shoes; but when they fitted these on, they never raised their eyes to look at their faces. For this the women conceived such a respect for them, that when they swore, they swore by the life of the holy Rabbis of the land of Israel." #@param {type:"string"}
paragraphs3 = [paragraph_1, paragraph_2, paragraph_3, paragraph_4, paragraph_5, paragraph_6, paragraph_7, paragraph_8]

paragraphs = np.concatenate([paragraphs1, paragraphs2, paragraphs3], axis = 0)

### Encoding the paragraphs
if transformer_name == 'universal-sentence-encoder':
    embeddings = embed(paragraphs)
    embeddings = embeddings.numpy()
else:
    embeddings = embed.encode(paragraphs)
    embeddings = np.array(embeddings)
print('The {} dimensional vector representation in the embedding space is: \n'.format(embeddings.shape[1]), embeddings)
print('The number of paragraphs is {}.'.format(embeddings.shape[0]))

### K-Elbow visualization - Silhouette Method. Good choice for the number of clusters is k that correspond to maximum of the Silhouette score.
# model = KMeans(random_state=0)
# visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=False)
# fig = visualizer.fit(embeddings)
# fig.show()
# st.set_option('deprecation.showPyplotGlobalUse', False)
# WCSS = []
# for i in range(2, 10):
#     model = KMeans(n_clusters=i, init='k-means++')
#     model.fit(embeddings)
#     WCSS.append(model.inertia_)
# fig = plt.figure(figsize=(7, 7))
# # fig.update_layout(
# #         autosize=True,
# #             margin=dict(
# #                 l=0,
# #                 r=0,
# #                 b=0,
# #                 t=40,
# #                 pad=0
# #             ),
# #         )
# plt.plot(range(2, 10), WCSS, linewidth=4, markersize=12, marker='o', color='red')
# plt.xticks(np.arange(2, 10))
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters=k).fit(embeddings)
  labels = kmeans.labels_
  sil.append(silhouette_score(embeddings, labels, metric='euclidean'))
plt.plot(range(2, kmax + 1), sil)
st.pyplot()

# ### Show clustering dendrogram. Alternative way to determine the number of clusters.
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(10, 7))
plt.title("Paragraphs dendrogram")
dend = shc.dendrogram(shc.linkage(embeddings, method='ward'))
st.pyplot()
#
# ### Unsupervised cluster labeling. Choose clustering method and the number of clusters (2-10)
# clustering_method = "k-means" #@param ["k-means", "hierarchical_clustering"]
# number_of_clusters =  2#@param {type:"integer"}
#
# if clustering_method == 'k-means':
#     km = KMeans(
#     n_clusters=number_of_clusters, init='random',
#     n_init=10, max_iter=300,
#     tol=1e-04, random_state=0)
#     y = km.fit_predict(embeddings)
#     print('Unsupervised paragraph labels: ', y)
# else:
#     cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='ward')
#     y = cluster.fit_predict(embeddings)
#
### Show the semantic paragraph similarity matrix.
# st.set_option('deprecation.showPyplotGlobalUse', False)
similarity_matrix = np.inner(embeddings, embeddings)
fig = plt.figure(1, figsize=(5, 5))
sns.set_palette("husl")
ax = sns.heatmap(similarity_matrix, annot = False, cmap ='YlGnBu',  linewidth=0.2)
#plt.show()
st.pyplot()
#
# # ### 3D PCA projection of the embedded paragraphs. Use mouse cursor to rotate the coordinate frame.
# # pca = PCA(n_components = 3)
# # pca.fit(embeddings)
# # X_pca = pca.transform(embeddings)
# #
# # tsne = TSNE(n_components=3)
# # X_tsne = tsne.fit_transform(embeddings)
# #
# # name = []
# # for i, _ in enumerate(X_pca):
# #     #name.append(str(i)+ '. ' + paragraphs[i][:20])
# #     name.append(str(i))
# # x, y, z = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
# #
# # fig = go.Figure(data=[go.Scatter3d(
# #     x=x,
# #     y=y,
# #     z=z,
# #     mode='markers+text',
# #     text = name,
# #     textposition="top center",
# #     marker=dict(
# #         size=5,
# #         color=y,
# #         colorscale='Viridis',
# #         opacity=0.8
# #     )
# # )])
# #
# # fig.update_layout(margin=dict(l=0.5, r=0, b=0, t=0.1))
# # fig.show()
# #
# #
# # ### 3D t-SNE projection of the embedded paragraphs. Use mouse cursor to rotate the coordinate frame.
# # pca = PCA(n_components = 3)
# # pca.fit(embeddings)
# # X_pca = pca.transform(embeddings)
# #
# # tsne = TSNE(n_components=3)
# # X_tsne = tsne.fit_transform(embeddings)
# #
# # name = []
# # for i, _ in enumerate(X_pca):
# #     #name.append(str(i)+ '. ' + paragraphs[i][:20])
# #     name.append(str(i))
# # x, y, z = X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2]
# #
# # fig = go.Figure(data=[go.Scatter3d(
# #     x=x,
# #     y=y,
# #     z=z,
# #     mode='markers+text',
# #     text = name,
# #     textposition="top center",
# #     marker=dict(
# #         size=5,
# #         color=y,
# #         colorscale='Viridis',
# #         opacity=0.8
# #     )
# # )])
# #
# # fig.update_layout(margin=dict(l=0.5, r=0, b=0, t=0.1))


### 3D UMAP projection of the embedded paragraphs. Use mouse cursor to rotate the coordinate frame.
reducer = umap.UMAP(n_components=3)
X_umap = reducer.fit_transform(embeddings)

pca = PCA(n_components = 3)
pca.fit(embeddings)
X_pca = pca.transform(embeddings)

name = []
for i, _ in enumerate(X_pca):
    #name.append(str(i)+ '. ' + paragraphs[i][:20])
    name.append(str(i))
x, y, z = X_umap[:, 0], X_umap[:, 1], X_umap[:, 2]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    text = name,
    textposition="top center",
    marker=dict(
        size=5,
        color=y,
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(margin=dict(l=0.5, r=0, b=0, t=0.1))

st.plotly_chart(fig)