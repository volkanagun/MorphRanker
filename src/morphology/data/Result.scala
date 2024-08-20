package morphology.data

import nn.embeddings.EmbedParams


case class Key(original: String, predicted: String)

case class Count(item: String, var count: Int) {
  def inc(): Count = {
    count += 1
    this
  }

  override def hashCode(): Int = item.hashCode()
  override def equals(obj: Any): Boolean = item.equals(obj.asInstanceOf[Count].item)
}


case class ResultMatrix() {
  val epsilon = 1E-10f
  var confusionMap = Map[Key, Float]()
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
  var tpsentence = 0f
  var cntsentence = 0f
  var cntanalysis = 0f
  var tptop5 = 0f

  def clear(): ResultMatrix = {
    confusionMap = Map[Key, Float]()
    originalMap = Map[String, Float]()
    predictionMap = Map[String, Float]()
    tpanalysis = 0f
    tpsentence = 0f
    cntanalysis = 0f
    tptop5 = 0f
    cntsentence = 0f
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

  def tagLabel(item:String):String= {
    "<LABEL ITEM=\"" + item + "\">"
  }

  def tagScore(label:String, score:Double):String= {
    "<TAG LABEL=\""+label+"\" SCORE=\""+score+"\"/>\n"
  }
  def itemScore(item:String, label:String, score:Double):String= {
    "<" + item+" LABEL=\""+label+"\" SCORE=\""+score+"\"/>\n"
  }

  def toXML(params: Params, filename: String): String = {

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

    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
      "<RESULTS EXPERIMENT = \"" + params.modelId() + "\">\n" +
      params.toXML() +
      "<DATASET  DATASET=\"" + filename + "\"/>\n" +
      "<SENTENCE TOTAL=\"" + cntsentence + "\"/>\n" +
      "<ANALYSIS TOTAL=\"" + cntanalysis + "\"/>\n" +
      "<SENTENCE TP=\"" + tpsentence + "\"/>\n" +
      "<SENTENCE ACCURACY=\"" + tpsentence / cntsentence + "\"/>\n" +
      "<ANALYSIS ACCURACY=\"" + tpanalysis / cntanalysis + "\"/>\n" +
      "<ANALYSIS TOP3=\"" + tptop5 / cntanalysis + "\"/>\n" +
      "<FMEASURE SCORE =\"" + fscore + "\"/>\n" +
      "<PRECISION SCORE =\"" + precisionScore + "\"/>\n" +
      "<RECALL SCORE =\"" + recallScore + "\"/>\n" +

      "<COVARIANCE>" +
      confusionMap.toArray.groupBy(_._1.original).toArray.map { case (original, scores) => {
        "<LABEL ITEM=\"" + original + "\">" +
          "<TAG LABEL=\"PRECISION\" SCORE=\"" + precisionMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"RECALL\" SCORE=\"" + recallMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"FMEASURE\" SCORE=\"" + fmeasureMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"TP\" SCORE=\"" + tpMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"FP\" SCORE=\"" + fpMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"FN\" SCORE=\"" + fnMap.getOrElse(original, 0f).toString + "\"/>\n" +
          "<TAG LABEL=\"TN\" SCORE=\"" + tnMap.getOrElse(original, 0f).toString + "\"/>\n" +
          scores.sortBy(pair => pair._2).map { case (key, score) => "<TAG ORIGINAL=\"" + key.original + "\"" + " PREDICTED=\"" + key.predicted + "\" COUNT=\"" + score.toString + "\"/>" }.mkString("\n") +
          "</LABEL>"
      }
      }.mkString("\n") +
      "</COVARIANCE>" +
      "</RESULTS>"
  }

  def increment(value: Boolean): ResultMatrix = {
    synchronized {
      tpsentence = tpsentence + (if (value) 1 else 0)
      cntsentence = cntsentence + 1
      this
    }
  }

  def add(word: Word, predicted: String): (Int, String) = {
    synchronized {
      val original = word.analyzes.head
      val foundinous = word.similarities(predicted, embedParams.secondOrder).reverse
        .take(5)
      val found = foundinous.last

      cntanalysis = cntanalysis + 1

      if (foundinous.contains(original)) {
        tptop5 = tptop5 + 1
      }

      if (original.equals(found)) {
        tpanalysis = tpanalysis + 1
        original.split(splitter.splitBy).tail.foreach(item => {
          val kk = Key(item, item)
          confusionMap = confusionMap.updated(kk, confusionMap.getOrElse(kk, 0f) + 1f)
        })
        (1, found)
      }
      else {
        val predictArray = found.split("\\+").tail
        original.split("\\+").tail.foreach(item => {
          predictArray.filter(pred => !pred.equals(item)).foreach(prog => {
            val kk = Key(item, prog)
            confusionMap = confusionMap.updated(kk, confusionMap.getOrElse(kk, 0f) + 1f)
          })
        })
        (0, found)
      }

    }
  }

  def addNoWord(word: Word, predicted: String): String = {
    synchronized {
      val foundinous = word.similarities(predicted, embedParams.secondOrder).reverse
        .take(5)
      val found = foundinous.last
      found
    }
  }
}
}
