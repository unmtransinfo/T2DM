from openai import OpenAI
import pandas as pd

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# read chart review file
infile = "/mnt/bigHDD/work/Grant/Diabetes/IOData/results_9/chart_review_data.tsv"

df = pd.read_csv(infile, sep="\t")
# person_ids = list(set(list(df.iloc[:,0])))
person_ids = [34286610401]

for j, p in enumerate(person_ids):
  p_df = df[df.iloc[:,0] == p]
  p_percentile = set(list(p_df.iloc[:,1]))
  p_features = set(list(p_df.iloc[:,2]))

  completion = client.chat.completions.create(
    extra_headers={
      "HTTP-Referer": "",
      "X-Title": "",
    },
    #model="openai/gpt-4.1",
    #model="anthropic/claude-3.7-sonnet",
    model="shisa-ai/shisa-v2-llama3.3-70b:free",
    #model="mistralai/mixtral-8x7b-instruct",
    #model="google/gemma-3-4b-it",
    #model="google/gemma-3-4b-it:free",
    max_tokens=250,
    messages=[
      {
        "role": "user",
        "content": "A person has the following list of conditions and medications: " + str(p_features) +" Use the list to tell what is the risk of T2DM for the person. Your answer should be - high to critical, moderate to high, mild to moderate, low to mild, and low. Do not give explanation"
      }
    ]
  )

  print(f"percentile: {p_percentile}, AI classification: {completion.choices[0].message.content}")



