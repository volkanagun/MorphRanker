package morphology.ranking

class RankWord(var rankMorphemes:Array[RankMorpheme]) {

  def copy():RankWord={
    new RankWord(rankMorphemes.map(_.copy()))
  }

  def reverse():RankWord={
    rankMorphemes = rankMorphemes.map(rankMorpheme=> rankMorpheme.reverseCopy())
    this
  }
}
