package morphology.ranking

class DataRank(var predictionIndex: Int, val items: Array[String], val score: Double) {
  //>> word1  tag1  tag2 >> word2  tag1  tag2  tag3
  val tokenBoundaries = items.mkString(" ").split(">>>").map(_.trim).filter(_.nonEmpty)

  def wordAnalysis(): Array[String] = {
    tokenBoundaries.map(wordAnalysis => wordAnalysis.split("\\s+").mkString("+"))
  }

  def wordAnalysisScores(): Array[DataRank] = {
    wordAnalysis().zipWithIndex.map(pair => {
      new DataRank(predictionIndex, pair._1.split("\\+"), score)
    })
  }

  def toAnalysis(): String = {
    items.mkString("+")
  }

}
