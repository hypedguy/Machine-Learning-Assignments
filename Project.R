library(tm)
library(ldatuning)
library(topicmodels)
library(LDAvis)
library(ggplot2)
library(SnowballC)
#Topic models
topicmodels_json_ldavis <- function(fitted, corpus, doc_term){
  # Required packages
  library(topicmodels)
  library(dplyr)
  library(stringi)
  library(tm)
  library(LDAvis)
  # Find required quantities
  phi <- posterior(fitted)$terms %>% as.matrix
  theta <- posterior(fitted)$topics %>% as.matrix
  vocab <- colnames(phi)
  doc_length <- vector()
  for (i in 1:length(corpus)) {
    temp <- paste(corpus[[i]]$content, collapse = ' ')
    doc_length <- c(doc_length, stri_count(temp, regex = '\\S+'))
  }
  temp_frequency <- inspect(doc_term)
  freq_matrix <- data.frame(ST = colnames(temp_frequency),
                            Freq = colSums(temp_frequency))
  rm(temp_frequency)
  # Convert to json
  json_lda <- LDAvis::createJSON(phi = phi, theta = theta,
                                 vocab = vocab,
                                 doc.length = doc_length,
                                 term.frequency = freq_matrix$Freq)
  return(json_lda)
}

setwd("/Users/virgiltataru/Desktop/PROJECT_ALL_LYRICS")
filenames <- list.files(getwd(),pattern="*.txt")
files <- lapply(filenames,readLines)
docs <- Corpus(VectorSource(files))

docs <-tm_map(docs,content_transformer(tolower))
docs <- tm_map(docs, content_transformer(removePunctuation))
docs <- tm_map(docs, content_transformer(removeNumbers))
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, content_transformer(stripWhitespace))
docs <- tm_map(docs, stemDocument)


dtm <- DocumentTermMatrix((docs))
rownames(dtm) <- filenames
freq <- colSums(as.matrix(dtm))
length(freq)
ord <- order(freq,decreasing=TRUE)
freq[ord]
write.csv(freq[ord],"word_freq.csv")
burnin <- 0
iter <- 200
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE
k <- 20
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm <- dtm[rowTotals> 0, ] 
ldaOut <-LDA(dtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs","_","DocsToTopics.csv"))
ldaOut.terms <- as.matrix(terms(ldaOut,5))
write.csv(ldaOut.terms,file=paste("LDAGibbs","_","TopicsToTerms.csv"))
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))
# result <- FindTopicsNumber(
#   dtm,
#   topics = seq(from = 2, to = 30, by = 1),
#   metrics = c("Griffiths2004", "CaoJuan2009"),
#   method = "Gibbs",
#   control = list(seed = 77),
#   mc.cores = 2L,
#   verbose = TRUE
# )
# FindTopicsNumber_plot(result)
serVis(topicmodels_json_ldavis(ldaOut, docs, dtm), out.dir = 'vis', open.browser = TRUE)


