package morphology.data

class DataStats {
  var totalTokens = 0d
  var totalTags = 0d
  var totalTagDepencies = 0d
  var totalAnalysis = 0d
  var totalDistantLink = 0
  var totalNonLocalLink = 0
  var totalLocalLink = 0
  var totalSentences = 0
  var totalAmbiguousSentences = 0
  var avgAmbiguouity = 0d


  def tag(name:String, value:Double):String={
    "<TAG LABEL=\"" + name + "\" SCORE=\""+value+"\"/>\n"
  }

  def toXML():String={
    "<STATS>\n" +
      tag("SENTENCE_COUNT", totalSentences) +
      tag("AMBIGUOUS_SENTENCE_COUNT", totalAmbiguousSentences) +
      tag("AMBIGUITY_AVG", avgAmbiguouity) +
      tag("TOKEN_COUNT", totalTokens) +
      tag("TAG_COUNT", totalTags) +
      tag("ANALYSIS_COUNT", totalAnalysis) +
      tag("AMBIGUITY_COUNT", totalAnalysis/totalTokens) +
      tag("LINK_COUNT", totalTagDepencies) +
      tag("DISTINCT_LINKS", totalDistantLink) +
      tag("NON_LOCAL_LINKS", totalNonLocalLink) +
      tag("LOCAL_LINKS", totalLocalLink) +
      "</STATS>"
  }

  def merge(stats: DataStats):this.type ={
    this.totalTokens += stats.totalTokens
    this.totalSentences += stats.totalSentences
    this.totalTags += stats.totalTags
    this.totalAnalysis += stats.totalAnalysis
    this.totalAmbiguousSentences += stats.totalAmbiguousSentences
    this.totalLocalLink += stats.totalLocalLink
    this.totalDistantLink += stats.totalDistantLink
    this.totalNonLocalLink += stats.totalNonLocalLink
    this.totalTagDepencies += stats.totalTagDepencies
    this
  }
}
