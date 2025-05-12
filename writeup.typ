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

+ See `cs336_data/language_identification.py`

+ If we're trying to train a model that performs well in a particular language, it would be highly problematic if, with high frequency, the language identifier (i) incorrecly classified other languages as the target language (ii) failed to recognize the target language. In the first case, the noisy data would prevent the model from learning the target language well. In the second case, the model might fail to be useful in some scenarios due to patchy understanding of the target language.

  In a high stakes scenario, we could (1) tune two confidence thresholds, discarding very low-confidence predictions ($c < C_"min"$), flagging moderate-confidence predictions ($C_"min" <= c < C_"max"$) for human-in-the-loop review, and accepting high-confidence predictions ($C_"max" <= c$). We could periodically audit and recalibrate the classifier by sampling predictions on held-out data. We could also ensemble multiple language‐ID models or supplement fastText with rule-based checks (e.g., keyword lists or script detection) to further reduce systematic errors.

+ In a sample of \~27k documents, the classifier reports that 43% are in English.

  The classifier seems to classify almost all documents in the same way that I would. However, when the classifier's score is low, the document often either (i) contains a mixture of languages (e.g. a restaurant menu with item names listed in English and Spanish), or (ii) is low-quality, containing a lot of random character strings, phone numbers, and the like. 

  Examples of both (i) and (ii) are shown below.

  From my observations, a threshold of \~0.85 would exclude the vast majority of ambiguous cases (no dominant language) and classification errors. The overwhelming majority of documents for which there is a clear correct classification have a score > 0.85, so we'd keep almost all high-quality documents, while discarding almost all others.

  Example of (i) (multiple languages causing low score):

  ```
  {
      "url": "https://grapevine.ca/listing/208-asper-trail-circle-ottawa-ontario-k2m-0k7-27802601/",
      "lang": "en",
      "score": 0.7182,
      "snippet": "| 613.829.1000   • Home   • Sell     • For Sale By Owner     • Fully Brokered     • Compare Services   • Buy     • Get Cash Back     • Grapevine Listings     • Ottawa Listings   • About Us     • Our Company     • Our Realtors     • Contact Us   • Sold & Saved     • Recent Sales     • Testimonials   LOADING    • « Go back  208 Asper Trail Circle Ottawa, Ontario K2M 0K7  view favourites   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 1 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 2 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 3 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 4 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 5 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 6 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 7 - X11923597   • 208 Asper Trail Circle, Ottawa, Ontario K2M 0K7 - Photo 8 - X11923597   • 208 Asper Trail "
  },
  ```

  Example of (ii) (low-quality content causing low score):

  ```
  {
      "url": "https://eventor.orientering.se/Ranking/ol/Event/Class/410195",
      "lang": "sv",
      "score": 0.7128,
      "snippet": "• In English In English  Produkter och tjänster    • Förbundssida   • Livesida   • Omaps   • Livelox   • hitta orientering Svenska Orienteringsförbundet Eventor - Svensk orienterings centrala IT-system   • Tävlingskalender   • Sverigelistan   • Pressresultat   • Forum   • Skapa konto   • Logga in   • Orientering   • Skidorientering   • Tävlingar     • Gallringsfilter     • Utlandstävlingar     • Betala rankingavgift     • Subjektiv ansökan   • Köp klubblicens   • Support och kontakt   • Frågor och svar  Sverigelistan, skog  Hela landetDistriktvisKlubbvis (Uppdaterad 2025-04-17)  Alla damer  Alla herrar  D21D20D18 D16 D35D40D45D50D55D60D65D70D75D80D85D90D95  H21H20H18 H16 H35H40H45H50H55H60H65H70H75H80H85H90H95  Rankingfilter, skog  Hela landet (Uppdaterat fredag 2025-04-11)  D21D20D18  H21H20H18  Sverigelistan, sprint  Hela landet (Uppdaterad 2025-04-17)  Alla damer  Alla herrar  D21D20D18 D16 D35D40D45D50D55D60D65D70D75D80D85D90D95  H21H20H18 H16 H35H40H45H50H55H60H65H70H75H80H85H90H9"
  }
  ```

== Problem (`mask_pii`): 3 points

+ See `cs336_data/mask_pii.py`

+ See `cs336_data/mask_pii.py`

+ See `cs336_data/mask_pii.py`

+ Issues with false positives: the language model may learn erroneous patterns in where phone numbers, emails, or IPs naturally occur in text, and may produce generations with these placholders in locations that appear nonsensical to a user.

  Issues with false negatives: failure to mask some PII in the training set could cause the language model to output real PII in its generations.

  Other issues: 
  - The model may refuse to produce generations that contain PII-like patterns. For example, a user might provide their email signature (containing PII) to a model and ask it to compose emails on the user's behalf. The model might then output emails ending with that signature, but with the PII masked, which would be frustrating for the user.
  - Not every email, phone number, or IP is PII that should be masked. For example, various helpline numbers or emails should probably not be masked.
  - Distribution shift: replacing diverse PII with uniform placeholders may cause the model to learn unnatural patterns and overgenerate these tokens.

  Mitigations:
  - Manually maintain a whitelist of PII-like data that should not be masked.
  - Use more structured or randomized placeholders (e.g. "|||EMAIL_ADDRESS_1|||" instead of "|||EMAIL_ADDRESS|||") to reduce the likelihood of overgeneration.

+ Some examples of false positives and false negatives are shown below.

  In general, email masking seems to be much more reliable than phone number masking, due to the more consistent and easily detectable structure of email addresses. Given the diversity of phone number formats, there are cases in which it's almost impossible to tell whether or not a string is a phone number without deep contextual understanding. 

  For example, in the first false positive listed below, a timestamp is masked as a phone number, simply because it's a string of digits of appropriate length. Only with some understanding of the surrounding context is this clear.

  False positive (emails masked, phone numbers missed):

  ```
  {
      "url": "http://indexrecruitment.com.np/testimonials",
      "text": "  • Enquiry\n  • Apply Now\n+977-1-5911443 |||EMAIL_ADDRESS|||\n+977-1-5911443 |||EMAIL_ADDRESS|||\n  • Home\n  • About Us\n    • Introduction\n    • Mission\n    • Vision\n    • Our Team\n    • Testimonials\n  • Services\n  • Gallery\n  • Blogs\n  • Contact\n  • Facebook\n  • Instagram\n  • Twitter\n  • Youtube\n\nWhat People Say About Us\n\nPrakash Aryal\n\nSteel Bender\n\nI haven’t worked here long but I can say this is a great place to work the management is very helpful and understanding and helped me with getting my friend on board same schedule\n\nABOUT OUR CONSULTING\n\nIndex Recruitment Pvt. Ltd.\n\n+977-1-5911443 Link\n\nQuick Links\n\n  • Home\n  • Team\n  • Our Gallery\n  • Contact Us\n  • Book An Appointment\n\nNEWSLETTER\n\n\n\n\nDesign & Developed By Web House Nepal",
      "emails_masked": 2,
      "phone_numbers_masked": 0,
      "ips_masked": 0,
      "all_masked": 2
  }  
  ```

  False negative (phone number missed):

  ```
  {
      "url": "http://adobsicimahi.org/undangan-rapat-pengurus-2016/",
      "text": "... Jenderal Achmad Yani (UNJANI)\nJl. Terusan Jenderal Gatot Subroto\nTelp. / Fax. |||PHONE_NUMBER|||\nHunting 0811 249 7890\n\n© 2016 ADOBSI.",
      "emails_masked": 0,
      "phone_numbers_masked": 1,
      "ips_masked": 0,
      "all_masked": 1
  }
  ```

  False positive (timestamp masked as phone number):

  ```
  {
      "url": "http://cdn.limetta.se/",
      "original_text": "imgix Configuration Details\n\nLast Deployed\nTue Mar 25, 2025 10:45:18 PM UTC (1742942718)\nHash\n\"1563\"\n\nDashboard Website",
      "text": "imgix Configuration Details\n\nLast Deployed\nTue Mar 25, 2025 10:45:18 PM UTC |||PHONE_NUMBER|||)\nHash\n\"1563\"\n\nDashboard Website",
      "emails_masked": 0,
      "phone_numbers_masked": 1,
      "ips_masked": 0,
      "all_masked": 1
  }
  ```

  Arguably a false positive (placeholder data masked as PII):

  ```
  {
      "url": "http://3rte.com.br/product/lixeira-com-tampa-35-litros-preta-plasvale/",
      "original_text": "About\n  • Services\n  • Contact\n  • Shop\n  • Cart\n  • Checkout\n  • My account\nContact Us\n1, My Address, My Street, New York City, NY, USA\n+1234567890\ncontact@domain.com\n1234567890\n© 2022 3RTE | PopularFX Theme",
      "text": "About\n  • Services\n  • Contact\n  • Shop\n  • Cart\n  • Checkout\n  • My account\nContact Us\n1, My Address, My Street, New York City, NY, USA\n+|||PHONE_NUMBER|||\n|||EMAIL_ADDRESS|||\n|||PHONE_NUMBER|||\n© 2022 3RTE | PopularFX Theme",
      "emails_masked": 1,
      "phone_numbers_masked": 2,
      "ips_masked": 0,
      "all_masked": 3
  }
  ```

== Problem (`harmful_content`): 6 points

+ See `cs336_data/harmful_content.py`

+ See `cs336_data/harmful_content.py`

+ Issues:

  Naive application of these filters could skew the training distribution. For example, though we may no longer use this language, it may be useful for a model to understand what a master-slave relationship refers to in a CS context. We may also lose culturally significant language that includes borderline profanity, but may be important for a model to understand. The classifiers may also simply produce false positives (removing benign content) and false negatives (retaining harmful content).

  Mitigations:
  - Calibrate confidence thresholds on manually vetted samples.
  - Use soft filtering or example reweighting rather than outright removal.
  - Incorporate periodic human-in-the-loop audits to refine classifier boundaries.

+ Examples of false positives and false negatives are shown below.

  In a collection of 26,820 documents, 72 were classified as NSFW (0.27%) and 224 were classified as toxic (0.84%).

  In general, finding an example of a false positive is much more difficult than finding a false negative, particularly for the NSFW classifier, for which there were many false negatives and almost no false positives. 

  No choice of thresholds will be perfect, because the classifiers' scores are very much imperfect (see the second example below, about which the NSFW classifier reports extremely high confidence for "non-NSFW").
  
  It seems high thresholds (e.g. requiring >0.9 for a "non-NSFW" or "non-toxic" classification) would be reasonable, given the rarity of false positives and frequency of false negatives, even when the classifier reports high confidence for the "non-NSFW" or "non-toxic" classification. Though this is subjective, there is probably some asymmetry here in that a small number of false negatives is likely much more harmful than a small number of false positives.

  When the classifiers are used together, it appears that a 0.9 threshold appears would catch a high proportion of all harmful content. For example, although the first example below should be classified as NSFW but isn't, it _is_ classified as toxic, and therefore would be removed.

  False positive (toxic):

  ```
  {
      "url": "http://www.w3schools.com/python/trypython.asp?filename=demo_default",
      "nsfw": "non-nsfw",
      "nsfw_score": 0.9975,
      "toxic": "toxic",
      "toxic_score": 0.7668,
      "snippet": "Get your own Python server ❯Run Code Ctrl+Alt+R Change Orientation Ctrl+Alt+O Change Theme Ctrl+Alt+D Go to Spaces Ctrl+Alt+P  Privacy policy and Copyright 1999-2025  Hello, World!"
  }
  ```

  False negative (should be NSFW; many such cases):

  ```
  {
      "url": "https://amaturepornmaster.com/homemade-porn-videos/cheerleader/",
      "nsfw": "non-nsfw",
      "nsfw_score": 0.9845,
      "toxic": "toxic",
      "toxic_score": 0.8715,
      "snippet": "Free Sextapes Porn Videos And HQ Home XXX Films Only On Amaturepornmaster.Com   • Top videos   • New videos   • All Models   • All Niches  Real Life Porn Tubes  Trinity Does What It Takes - Trinity May Trinity Does What It Takes - Trinity May Candice Delaware In Exotic Xxx Movie Upskirt Watch , Its Amazing Candice Delaware In Exotic Xxx Movie Upskirt Watch , Its Amazing From To Cum Dumpster - Scarlett Mae From To Cum Dumpster - Scarlett Mae Raising Your Spirit With Nia Nacci Raising Your Spirit With Nia Nacci Big Butt Cheerleader Does Splits On The Dick - Belle Sparkles Big Butt Cheerleader Does Splits On The Dick - Belle Sparkles Beach Blonde Cheerleader Paisley Porter Takes That Huge Prick Deep Into Her Tight Vagina Beach Blonde Cheerleader Paisley Porter Takes That Huge Prick Deep Into Her Tight Vagina Crazy Porn Movie Big Tits Watch , It's Amazing Crazy Porn Movie Big Tits Watch , It's Amazing Jazzi Lai - Black Cheerleader Gang 26 Jazzi Lai - Black Cheerleader Gang 26 A Petite With"
  }
  ```

== Problem (`gopher_quality_filters`): 3 points

+ See `cs336_data/gopher_quality_filters.py`

+ On a sample of 26,820 documents, 7,362 passed the quality filters (27.45%). 

  I generally found high agreement between the quality filters and my own judgement.

  Cases of mild disagreement:
  
  There were instances in which the filters rejected documents that I might have kept, like the following product page: https://www.footballgiftsonline.co.uk/products/everton-pulse-double-duvet-set. There's certainly high-quality training data on this page, but perhaps if we're only applying filters at the page level (rather than trying to extract the prose from the page), perhaps discarding it is more reasonable.

  It's not obvious though, why that product page was filtered out, but this one was not: https://www.dshirt14.it/en/t-shirt/6988-T-shirt-Kids-Pop-Origami-Scottish-Terrier.html. If anything the former appears more useful as training data, though neither is particularly dense with worthwhile text, so perhaps with page-level filtering both should be discarded.

  There were may such cases of ecommerce-related pages that seem to be close to the boundary, as it's not clear why some are kept and other aren't.

  Case of strong disagreement:

  Though rare, there were a few cases, such as the one below, where I certainly would have kept a page that the filters rejected. Below is one example:

  https://www.hji.edu/being-a-peacemaker-leadership-series-event/

== Problem (`quality_classifier`): 15 points

+ See `cs336_data/quality_classifier.py`

+ See `cs336_data/quality_classifier.py`

== Problem (`exact_deduplication`): 3 points

+ See `cs336_data/exact_deduplication.py`