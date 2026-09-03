import os
import json
import pandas as pd
from openai import OpenAI

# ---------------------------------------
# 1. Initialize OpenAI client
# ---------------------------------------
# Grabs the key out of your active environment payload
client = OpenAI()

# ---------------------------------------
# 2. Load CSV data
# ---------------------------------------
csv_path = "data/transcriptions.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing file: {csv_path}. Please place your transcription file in the data/ directory.")

df = pd.read_csv(csv_path)

print(f"Loaded dataset successfully. Total records: {len(df)}")
print(df.head(2))


# ---------------------------------------
# 3. Function to process one transcription
# ---------------------------------------
def extract_medical_data(transcription, medical_specialty):
    """
    Extract age, treatment/procedure and ICD code
    from a medical transcription using one OpenAI tool calling execution.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a healthcare data extraction assistant.\n\n"
                "Extract structured information from the medical transcription.\n\n"
                "Return:\n"
                "1. Patient age (as an integer number if present, or -1 if completely unknown)\n"
                "2. Recommended treatment or procedure\n"
                "3. Suggested ICD-10-CM code related to the recommended treatment, procedure, or medical condition.\n\n"
                "If text content or metrics are missing, supply 'Unknown'.\n\n"
                "The ICD code should be treated as an AI-generated suggestion and should be verified by a qualified medical coding professional."
            )
        },
        {
            "role": "user",
            "content": f"Medical Specialty:\n{medical_specialty}\n\nMedical Transcription:\n{transcription}\n\nExtract the requested medical information."
        }
    ]

    # ---------------------------------------
    # Functional Tool Call Parameter Setup
    # ---------------------------------------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_medical_data",
                "description": "Extract structured medical fields from plaintext files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Age": {
                            "type": "integer",
                            "description": "Patient's age as a numeric digit. Use -1 if absent."
                        },
                        "Recommended Treatment/Procedure": {
                            "type": "string",
                            "description": "Explicit recommended treatment, therapeutic prescription, or surgical procedure."
                        },
                        "ICD Code": {
                            "type": "string",
                            "description": "Suggested structural ICD-10-CM code alphanumeric layout mapping directly to the diagnostic evaluation."
                        }
                    },
                    "required": [
                        "Age",
                        "Recommended Treatment/Procedure",
                        "ICD Code"
                    ]
                }
            }
        }
    ]

    # Invoke chat engine completion targeting specific tool function schemas
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "extract_medical_data"}
        }
    )

    # Dissect tool call arguments safely
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = tool_call.function.arguments

    return json.loads(arguments)


# ---------------------------------------
# 4. Process the dataset
# ---------------------------------------
processed_data = []

for index, row in df.iterrows():
    print(f"Processing record {index + 1}/{len(df)}...")

    try:
        med_specialty = row["medical_specialty"]
        raw_transcription = row["transcription"]

        # Call OpenAI Function Pipeline
        extracted_data = extract_medical_data(raw_transcription, med_specialty)

        # Standardize structural object fields for pandas ingestion
        extracted_data["Medical Specialty"] = med_specialty
        extracted_data["Transcription"] = raw_transcription

        processed_data.append(extracted_data)

    except Exception as e:
        print(f"⚠️ Error parsing index line row {index}: {e}")
        # Insert a placeholder row to keep the index aligned if an explicit failure happens
        processed_data.append({
            "Age": -1,
            "Recommended Treatment/Procedure": "Error / Unknown",
            "ICD Code": "Unknown",
            "Medical Specialty": row.get("medical_specialty", "Unknown"),
            "Transcription": row.get("transcription", "")
        })


# ---------------------------------------
# 5. Create structured DataFrame
# ---------------------------------------
df_structured = pd.DataFrame(processed_data)


# ---------------------------------------
# 6. Display final result console logs
# ---------------------------------------
print("\n" + "="*80 + "\nFINAL STRUCTURED DATA DATAFRAME\n" + "="*80)
print(df_structured.head().to_string())


# ---------------------------------------
# 7. Save results out to disk
# ---------------------------------------
output_file = "data/structured_medical_data.csv"
df_structured.to_csv(output_file, index=False)

print(f"\nProcessing Complete! Cleaned matrices successfully exported to: {output_file}")
