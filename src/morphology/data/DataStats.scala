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


  def tag(name:String, value:Double):String={
    "<TAG LABEL=\"" + name + "\" SCORE=\""+value+"\"/>\n"
  }

  def toXML():String={
    "<STATS>\n" +
      tag("SENTENCE_COUNT", totalSentences) +
      tag("AMBIGUOUS_SENTENCE_COUNT", totalAmbiguousSentences) +
      tag("TOKEN_COUNT", totalTokens) +
      tag("TAG_COUNT", totalTags) +
      tag("ANALYSIS_COUNT", totalAnalysis) +
      tag("LINK_COUNT", totalTagDepencies) +
      tag("DISTINCT_LINKS", totalDistantLink) +
      tag("NON_LOCAL_LINKS", totalNonLocalLink) +
      tag("LOCAL_LINKS", totalLocalLink) +
      "</STATS>"
  }
}
