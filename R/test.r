rm(list=ls())
library("SparseM")
library("Matrix")
setwd("~/workspace_C/CPP_BIO")
enose  <- read.table("EnoseAllSamples.csv", sep=",", header=TRUE, row.names=1)
sensory<-read.table("SensoryAllSamples.csv", header = TRUE, sep =",",row.names=1)

svm.training <-
  function (enose,sensory) {
dyn.load(paste("library.so"))
rows=length(enose[1,])
cols=length(enose[,1])
enose=as.matrix(enose)
enose <- SparseM::t(enose)
enose=as.double(enose)
sensory = as.matrix(sensory)
sensory = as.double(sensory)
out <- .C ("do_training1",as.integer(rows),as.integer(cols),as.double(enose),as.double(sensory),test = as.double(1000),test1 = as.double(1000))
dyn.unload(paste("library.so"))
out
}

o1 = svm.training(enose,sensory)