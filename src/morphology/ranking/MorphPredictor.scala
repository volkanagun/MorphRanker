package morphology.ranking

import morphology.data.{DataStats, Dataset, MorphRanker, Params, Sentence}
import morphology.morph.{MorphAnalyzer, Tokenizer}

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

abstract class MorphPredictor(val params:Params) extends Serializable{

  val tokenizer = new Tokenizer()
  val analyzer = new MorphAnalyzer()
    .setAmbiguityFreq(params.maxPairAmbiguity)
  val stats = new DataStats()
  var order: Array[Int] = Array[Int](1, 2, 3)
  var window: Int = 20
  var isTraining = true

  def setOrder(order: Array[Int]): MorphPredictor = {
    this.order = order
    this
  }

  def setWindow(window: Int): MorphPredictor = {
    this.window = window
    this
  }


  def setIsTraining(value: Boolean): MorphPredictor = {
    isTraining = value
    this
  }


  def load(): MorphPredictor = {
    val modelFilename = params.rankModelFilename()
    if (new File(modelFilename).exists()) {
      println("Loading model")
      val objectStream = new ObjectInputStream(new FileInputStream(modelFilename))
      load(objectStream)
    }
    this
  }

  def save(): MorphPredictor = {
    println("Saving model...")
    val objectStream = new ObjectOutputStream(new FileOutputStream(params.rankModelFilename()))
    save(objectStream)
    objectStream.close()
    this
  }

  def setAmbiguityFreq(freq: Int): MorphPredictor = {
    analyzer.setAmbiguityFreq(freq)
    this
  }

  def statistics(sentence: Sentence): Unit = {
    stats.totalSentences += 1
    stats.totalAnalysis += sentence.countAnalysis()
    stats.totalTokens += sentence.length
    stats.totalTags += sentence.countTags()
  }

  def trigger():MorphPredictor

  def train(sentences: Array[String]): MorphPredictor = {
    println("Parsing sentences")
    sentences.par.map(sentenceLine=>{
      Dataset.parseSentence(analyzer, sentenceLine)
    }).toArray.foreach(sentence => {
      val sentenceAmbiguity = sentence.countAmbiguity()
      if (sentenceAmbiguity <= params.maxSentenceAmbiguity) {
        statistics(sentence)
        construct(sentence)
      }
    })

    this
  }

  def save(objectStream: ObjectOutputStream): MorphPredictor
  def load(objectStream: ObjectInputStream): MorphPredictor
  def infer(results: Array[RankWord]): Map[Int, Array[RankMorpheme]]

  def construct(sequence: Sentence): MorphPredictor
  def merge(other: MorphPredictor):MorphPredictor
}
