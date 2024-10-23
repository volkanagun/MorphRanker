package morphology.data


case class Key(original: String, predicted: String)

case class Count(item: String, var count: Double) {
  def inc(): Count = {
    count += 1
    this
  }
  def incBy(value:Double): Count = {
    count += value
    this
  }


  override def hashCode(): Int = item.hashCode()

  override def equals(obj: Any): Boolean = item.equals(obj.asInstanceOf[Count].item)
}


case class Result(params: Params) {
  val epsilon = 1E-10f
  var confusionMap = Map[Key, Float]()
  var worstCaseMap = Map[Key, Float]()

  var originalMap = Map[String, Float]()
  var sumCount = 0f


  var tpMap = Map[String, Float]()
  var fpMap = Map[String, Float]()
  var fnMap = Map[String, Float]()
  var tnMap = Map[String, Float]()
  var precisionMap = Map[String, Float]()
  var recallMap = Map[String, Float]()
  var fmeasureMap = Map[String, Float]()

  var predictionMap = Map[String, Float]()
  var tpanalysis = 0f
  var tpambiguous = 0f
  var tptop5 = 0f
  var countAmbigous = 0f
  var tpsentence = 0f
  var countSentence = 0f
  var countAnalysis = 0f

  def clear(): Result = {
    confusionMap = Map[Key, Float]()
    originalMap = Map[String, Float]()
    predictionMap = Map[String, Float]()
    tpanalysis = 0f
    tpsentence = 0f
    countAnalysis = 0f
    tptop5 = 0f
    countSentence = 0f
    this
  }

  def originalCount(): Map[String, Float] = {
    confusionMap.toArray.map(pair => (pair._1.original, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toMap
  }

  def totalCount(): Float = {
    confusionMap.toArray.map(pair => (pair._1.original, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toArray.map(_._2).sum
  }



  def tpCount(): Map[String, Float] = {
    confusionMap.toArray.filter(pair => pair._1.predicted.equals(pair._1.original)).map(pair => (pair._1.original, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toMap
  }

  def fpCount(): Map[String, Float] = {
    confusionMap.toArray.filter(pair => !pair._1.predicted.equals(pair._1.original)).map(pair => (pair._1.predicted, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toMap
  }

  def fnCount(): Map[String, Float] = {
    confusionMap.toArray.filter(pair => !pair._1.predicted.equals(pair._1.original)).map(pair => (pair._1.original, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toMap
  }

  def tnCount(): Map[String, Float] = {
    originalMap.map { case (original, count) => (original, sumCount - tpMap.getOrElse(original, epsilon) - fpMap.getOrElse(original, epsilon) - fnMap.getOrElse(original, epsilon)) }
  }

  def predictionCount(): Map[String, Float] = {
    confusionMap.toArray.map(pair => (pair._1.predicted, pair._2)).groupBy(_._1).view.mapValues(_.map(_._2).sum)
      .toMap
  }

  def precision(): Map[String, Float] = {
    originalMap.map { case (key, score) => (key, tpMap.getOrElse(key, epsilon) / (fpMap.getOrElse(key, epsilon) + tpMap.getOrElse(key, epsilon))) }
  }

  def recall(): Map[String, Float] = {
    originalMap.map { case (key, score) => (key, tpMap.getOrElse(key, epsilon) / (fnMap.getOrElse(key, epsilon) + tpMap.getOrElse(key, epsilon))) }
  }

  def fmeasure(): Map[String, Float] = {
    precisionMap.map { case (key, score) => (key, 2 * score * recallMap.getOrElse(key, 1f) / (recallMap.getOrElse(key, epsilon) + score)) }
  }

  def tagLabel(item: String): String = {
    "<LABEL ITEM=\"" + item + "\">"
  }

  def tagScore(label: String, score: Float): String = {
    "<TAG LABEL=\"" + label + "\" SCORE=\"" + score + "\"/>\n"
  }

  def itemScore(label: String, score: Float): String = {
    "<" + label + " SCORE=\"" + score + "\"/>\n"
  }

  def counts(array: Array[(Key, Float)]): String = {
    array.sortBy(pair => pair._2).map { case (key, score) => "<TAG ORIGINAL=\"" + key.original + "\"" + " PREDICTED=\"" + key.predicted + "\" COUNT=\"" + score.toString + "\"/>" }.mkString("\n")
  }

  def toShortLine():String = {
    val analysisAcc = tpanalysis / countAnalysis
    val ambiguousAcc = tpambiguous / countAmbigous
    val topacc = tptop5 / countAnalysis
    params.skipHeadAmbiguity + "," + params.maxSliceNgram + "," + params.prunningRatio + "," + params.maxTokenWindow + "," + analysisAcc + "," + ambiguousAcc + "," + topacc
  }

  def toXML(filename: String): String = {

    sumCount = totalCount()
    originalMap = originalCount()
    predictionMap = predictionCount()
    tpMap = tpCount()
    fpMap = fpCount()
    fnMap = fnCount()
    tnMap = tnCount()
    precisionMap = precision()
    recallMap = recall()
    fmeasureMap = fmeasure()

    val fscore = fmeasureMap.map(_._2).sum / fmeasureMap.size
    val precisionScore = precisionMap.map(_._2).sum / precisionMap.size
    val recallScore = recallMap.map(_._2).sum / recallMap.size

    val sentenceAcc = tpsentence / countSentence
    val analysisAcc = tpanalysis / countAnalysis
    val analysisTop5Acc = tptop5 / countAnalysis
    val worstCases = worstCaseMap.toArray.sortBy(pair=> pair._2).reverse.take(50)

    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
      "<RESULTS EXPERIMENT = \"" + params.modelID() + "\">\n" +
      params.toXML() +
      "<DATASET  DATASET=\"" + filename + "\"/>\n" +
      "<SENTENCE TOTAL=\"" + countSentence + "\"/>\n" +
      "<ANALYSIS TOTAL=\"" + countAnalysis + "\"/>\n" +
      "<SENTENCE TP=\"" + tpsentence + "\"/>\n" +
      "<SENTENCE ACCURACY=\"" + sentenceAcc + "\"/>\n" +
      "<ANALYSIS ACCURACY=\"" + analysisAcc + "\"/>\n" +
      "<AMBIGUOUS ACCURACY=\"" + tpambiguous/countAmbigous + "\"/>\n" +
      "<ANALYSIS TOP5=\"" + analysisTop5Acc  + "\"/>\n" +
      itemScore("FMEASURE", fscore) +
      itemScore("PRECISION", precisionScore) +
      itemScore("RECALL", recallScore) +
      "<WORST_CASES>\n" +
      worstCases.map(pair=> "<ITEM ORIGINAL=\"" + pair._1.original + "\" PREDICTED=\""+pair._1.predicted+"\"/>").mkString("\n") +
      "</WORST_CASES>\n" +
      "<COVARIANCE>\n" +
      confusionMap.toArray.groupBy(_._1.original).toArray.map { case (original, scores) => {
        "<LABEL ITEM=\"" + original + "\">\n" +
          tagScore("PRECISION", precisionMap.getOrElse(original, 0f)) +
          tagScore("RECALL", recallMap.getOrElse(original, 0f)) +
          tagScore("FMEASURE", fmeasureMap.getOrElse(original, 0f)) +
          tagScore("TRUEPOSITIVES", tpMap.getOrElse(original, 0f)) +
          tagScore("FALSEPOSITIVES", fpMap.getOrElse(original, 0f)) +
          tagScore("TRUENEGATIVES", tnMap.getOrElse(original, 0f)) +
          tagScore("FALSENEGATIVES", fnMap.getOrElse(original, 0f)) +
          counts(scores) + "\n" +
          "</LABEL>"
      }
      }.mkString("\n") + "\n" +
      "</COVARIANCE>" +
      "</RESULTS>"
  }

  def incrementTop5(count:Int):Result={
    tptop5  = count
    this
  }

  def incrementAmbiguous(totalAmbiguous:Int, trueAmbigous:Int):Result={
    this.countAmbigous += totalAmbiguous
    this.tpambiguous += trueAmbigous
    this
  }

  def increment(value: Boolean): Result = {
    tpsentence = tpsentence + (if (value) 1 else 0)
    countSentence = countSentence + 1
    this
  }

  def addTop5(word:Word, predicted:Array[String]):Int={
    val trueLabel = word.originalAnalyses.head
    if(predicted.contains(trueLabel)){
      1
    }
    else {
      0
    }
  }

  def addAmbiguous(word:Word, predicted:String):(Int, Int)={
    val trueLabel = word.originalAnalyses.head
    if(!word.notAmbiguous()){
      if(trueLabel.equals(predicted)) (1, 1)
      else (1, 0)
    }
    else{
      (0, 0)
    }
  }

  def addPrediction(word: Word, predicted: String): (Int, String) = {

    val original = word.originalAnalyses.head
    if(original.equals(predicted)){
      tpanalysis = tpanalysis + 1
      countAnalysis = countAnalysis + 1
      original.split(word.splitBy).tail.foreach(item => {
        val kk = Key(item, item)
        confusionMap = confusionMap.updated(kk, confusionMap.getOrElse(kk, 0f) + 1f)
      })
      (1, original)
    }
    else{

      val predictMorphtags = predicted.split(word.splitBy).tail
      val originalMorphtags = original.split(word.splitBy).tail
      originalMorphtags.foreach(originalTag => {
        val falsePredictions = predictMorphtags.filter(pred => !pred.equals(originalTag))
        falsePredictions.foreach(predictedTag => {
          val confusionKey = Key(originalTag, predictedTag)
          confusionMap = confusionMap.updated(confusionKey, confusionMap.getOrElse(confusionKey, 0f) + 1f)
        })
      })
      countAnalysis = countAnalysis + 1

      val worstKey = Key(original, predicted)
      worstCaseMap = worstCaseMap.updated(worstKey, worstCaseMap.getOrElse(worstKey, 0f) + 1f)

      (0, predicted)
    }
  }



  def mostSimilar(word: Word, predicted: String): String = {
    val foundinous = word.similarity(predicted)
    val found = foundinous.last
    found
  }

  def merge(other: Result): Result = {
    countAnalysis += other.countAnalysis
    countSentence += other.countSentence

    countAmbigous += other.countAmbigous
    tpambiguous += other.tpambiguous
    tptop5 += other.tptop5
    tpanalysis += other.tpanalysis
    tpsentence += other.tpsentence
    other.confusionMap.foreach { case (key, count) => {
      confusionMap = confusionMap.updated(key, confusionMap.getOrElse(key, 0f) + count)
    }}

    other.worstCaseMap.foreach { case (key, count) => {
      worstCaseMap = worstCaseMap.updated(key, worstCaseMap.getOrElse(key, 0f) + count)
    }}

    this
  }

}
