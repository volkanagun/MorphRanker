package morphology.ranking

case class RankSequence(var rankMorphemes: Array[RankMorpheme], var targetIndex:Int, var score:Double = 0d) {

  def setTargetIndex(index:Int):this.type ={
    this.targetIndex = index
    this
  }

  def setScore(score:Double):this.type ={
    this.score = score
    this
  }

  def addScore(score:Double):this.type ={
    this.score += score
    this
  }

 /* def reverseCopy():RankSequence={
    RankSequence(rankMorphemes.reverse.map(_.reverseCopy()), targetIndex, score)
  }*/

  def reverse():this.type ={
    rankMorphemes = rankMorphemes.reverse.map(_.reverseCopy())
    this
  }


  def getRankMorpheme(index:Int):RankMorpheme = rankMorphemes(index)
  def getRankMorpheme():RankMorpheme = rankMorphemes(targetIndex)

  def copy():RankSequence={
    RankSequence(rankMorphemes.map(_.copy()), targetIndex, score)
  }
}
