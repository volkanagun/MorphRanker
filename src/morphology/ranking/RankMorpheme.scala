package morphology.ranking

import morphology.data.Label

case class RankMorpheme(var tokenIndex: Int, var analysis: String, var labels: Array[Label], var count: Double, var rank: Double) extends Ordered[RankMorpheme] {

  var negativeLogLikelihood = 0d

  override def compare(that: RankMorpheme): Int = {
    if(that.rank == rank) this.negativeLogLikelihood.compare(that.negativeLogLikelihood)
    else this.rank.compare(that.rank)
  }

  def setRank(score: Double): this.type = {
    this.rank = score
    this
  }
  def setLoglikelihood(score: Double): this.type = {
    this.negativeLogLikelihood = score
    this
  }

  def rankBy(tags:Array[String]):Double ={
    val size = labels.length * tags.length
    val total = tags.map(tag=> labels.filter(label=> label.tags.contains(tag)).length).sum
    total.toDouble / size
  }


  def incRank(score: Double): this.type = {
    this.rank += score
    this
  }

  def ambiguityScore(): Double = {
    labels.head.ambiguousScore()
  }

  def lessAmbiguous(other:RankMorpheme):Boolean={
    ambiguityScore() < other.ambiguityScore()
  }

  def notAmbiguous(): Boolean = {
    ambiguityScore() == 1.0
  }

  def reverseCopy(): RankMorpheme = {
    val newLabels = this.labels.reverse
    RankMorpheme(tokenIndex, analysis, newLabels, count, rank)
  }

  def reverse(): this.type = {
    this.labels = this.labels.reverse
    this
  }

  def addRank(score: Double): this.type = {
    this.rank += score
    this
  }

  override def hashCode(): Int = analysis.hashCode() * 7 + tokenIndex

  override def equals(obj: Any): Boolean = {
    val otherMorpheme = obj.asInstanceOf[RankMorpheme]
    otherMorpheme.analysis.equals(analysis) && otherMorpheme.tokenIndex == tokenIndex
  }

  override def toString: String = analysis

  def copy(): RankMorpheme = {
    RankMorpheme(tokenIndex, analysis, labels, count, rank)
  }
}
