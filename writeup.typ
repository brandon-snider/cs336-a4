#import "@preview/ilm:1.4.1": *
#import "@preview/tablem:0.2.0": *

#let three-line-table = tablem.with(
  render: (columns: auto, ..args) => {
    table(
      columns: columns,
      stroke: none,
      align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 4],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 05, day: 20),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
  bibliography: bibliography("refs.bib")
)

#set enum(numbering: "a)")
#set heading(numbering: none)
#show link: underline

#set table(
    inset: 6pt, // default is 5pt
    stroke: (0.5pt + stroke-color),
)

= 2 Filtering Common Crawl

== Problem (`look_at_cc`): 4 points

+ URL: http://0371rykj.com/ipfhsb/34.html

  The URL is no longer accessible.

  It looks like the page could be an e-commerce website or profile page for a company called "Shanghai Linpin Instrument Stock Co Ltd", but that's about as much as I can gather from the raw HTML, most of which is not in English.

+ The extracted text appears to contain URL paths, phone numbers, unique identifiers of some sort (e.g. product codes), product specifications, and possible a category selector or filtering mechanism of some sort (along with a lot of non-English text).

  At a minimum, information like phone numbers and email addresses should be filtered out. Even if people or companies want this information published, they may not want it embedded in a model trained on this data. Additionally, having content that is structured like a web page could cause a model trained on this data to learn patterns in the layouts and content of web pages, which may not be desirable depending on what the model is being trained for (e.g. answering questions, as opposed to generating web-style content).

  That said, a page like this could help a model learn about the structure of webpages, or about product specifications and their relationship to surrounding content, which could be useful depending on the use case.

+ If our use case is a copilot for writing optimized product pages, it could be beneficial to include content like product specifications and descriptions, structured like a web page, in the training data. If we're building an AI therapist, this data is unlikely to be useful.

+ Examples until first high-quality page: 3

  The table below shows my findings in the first 25 contentful results from the Common Crawl sample. Almost all of them would ideally be filtered out. I happened to find a page containing some high-quality content at the third position (a page for a computational mechanics olympiad containing some fluent English), but this was the only such page in the sample.

  #figure(tablem[
  | Language                  | Page Type                   | Domain Name                    |
  |---------------------------|-----------------------------|-------------------------------|
  | Chinese                   | E-commerce/Industrial       | 0371rykj.com                  |
  | Chinese, English          | Fan Blog/Forum              | 10www.chinatikfans.com        |
  | English                   | Conference/Academic         | 13.usnccm.org                 |
  | Chinese                   | Adult Chat                  | 176.utchat888.com             |
  | Japanese, Chinese         | Spam/Error Page             | 176766.cn                     |
  | Chinese                   | Spam/Error Page             | 178mh.com                     |
  | Chinese                   | Adult Live Stream           | 1796370.tgtg97.com            |
  | Chinese, English          | Adult Video                 | 18sex.v340.info               |
  | Dutch, English            | Blog                        | 1kb.klimtoren.be              |
  | Greek, English            | Help Desk/Forum             | 1pekesat-exae.mysch.gr        |
  | Greek, English            | Help Desk/Forum             | 1pekesat-exae.mysch.gr        |
  | Chinese                   | Content Farm/App            | 1s6605084.yhxzseo.com         |
  | Turkish, Danish, English  | Software Doc                | 20com20.fr                    |
  | English                   | Gambling                    | 24ktcasino.net                |
  | English                   | Error Page                  | 2kgames.eu                    |
  | Chinese                   | Content Farm/App            | 2l6185919.yizhangting.com     |
  | Chinese                   | Spam                        | 303323.com                    |
  | Chinese                   | Pirate Streaming/Anime      | 30bad.com                     |
  | Chinese                   | Unclear                     | 312001.net                    |
  | Chinese                   | Adult Chat/Dating           | 354577.mwe075.com             |
  | English                   | Empty Search Results        | 356.schoollibrary.edu.pe.ca   |
  | Chinese                   | Adult Chat/Video            | 366392.haaxz.com              |
  | Chinese, English          | Adult Chat/Dating           | 387tel.com                    |
  | Spanish                   | Blog/Forum                  | 3diasdemarzo.blogspot.com     |
  ], caption: "First 25 WET records from the Common Crawl sample")

== Problem (`extract_text`): 3 points

+ See `cs336_data/extract_text.py`

+ In general, the extracted text in the WET files is much higher-quality and that produced by the custom function. The custom function's extracted text is mostly whitespace, and contains some formatting characters (like bullet points). Both the whitespace and bullet points have been stripped from the WET files' extracted text. There are also some substantive content differences; for example, the custom function's text contains image names like "bridge_USACM_trim.jpg", which are not present in the WET files.

== Problem (`language_identification`): 6 points

+ TODO

+ TODO

+ TODO