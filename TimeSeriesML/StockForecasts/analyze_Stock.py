import os
import openai
import sys

# Example prompt
# "In 2021, is Tesla (TSLA) a good stock investment? No, semiconductor supply chain shortages are impacting the ability to generate revenue.\nIn 2021, is Ford (F) a good stock investment? Yes, electric vehicles sales are at an all time high and growing.\nIn 2021, is The One Group (STKS) a good stock investment? Yes, hospitality growth is growing with strong revenues.\nIn 2021, is Ethereum Classic Investment Trust (ETCG) a good stock investment? Yes, decentralized finance is a growing global trend and liquidity is increasing.\nIn 2021, is Apple (AAPL) a good stock investment? No, the company is underperforming the market despite constant positive press because of a stretched valuation coupled with challenging future expectations.\nIn 2021, is BLADE AIR MOBILITY INC (BLDE) a good stock investment?"

if len(sys.argv) > 1:
    promptText = sys.argv[1]
else:
    promptText = "Error"
    
if promptText != "Error":

    # Load your API key from an environment variable or secret management service
    openai.api_key = "apikey"

    response = openai.Completion.create(engine="davinci-instruct-beta", prompt=promptText, temperature=0.4, top_p=1, frequency_penalty=0.2, stop=["\\n"], max_tokens=120)
    print("AIJustificationStart")
    print(response.choices[0]["text"])
    print("AIJustificationEnd")


