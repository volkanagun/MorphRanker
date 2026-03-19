package morphology.experiment

import morphology.data.{MorphRanker, Params}

import java.io.PrintWriter
import scala.collection.parallel.CollectionConverters.ImmutableIterableIsParallelizable
import scala.io.Source

object MorphDataset {

  val params = initParams()
  var experiments = new MorphExperiment(params)

  def initParams(): Params = {
    val params = new Params()
    params.model = "rank"
    params.prunningRatio = 0.25
    params.maxPairAmbiguity = 1
    params.skipHeadAmbiguity = 3
    params.maxTokenWindow = 7
    params.maxSentenceAmbiguity = Int.MaxValue
    params.maxSentences = 1000
    params.maxRankIters = 10
    params.maxEpocs = 100
    params.maxSliceNgram = 3
    params.sentenceFilename = "resources/sentences/sentences-april-v2-tr.txt"
    params
  }

  def train():Unit = {
    experiments.constructTraining().train()
  }

  def predictor():MorphRanker = {
    new MorphRanker(params)
  }

  def createLineByLine():Unit = {
    val morphRanker = predictor()
    val pw = new PrintWriter(params.lineByLineAnnotation)
    Source.fromFile(params.sentenceFilename).getLines().sliding(params.threadSize, params.threadSize)
      .foreach(sentenceList => {
        val pairs = sentenceList.map(sentence=> (sentence, morphRanker.infer(sentence)))
          .filter(pair=> pair._2.isDefined).toArray
        for ((sentence, analyses) <- pairs){
          pw.println(sentence)
          pw.println(analyses)
        }
      })
    pw.close()
  }

  def main(args: Array[String]): Unit = {
    //train()
    createLineByLine()
  }

}
