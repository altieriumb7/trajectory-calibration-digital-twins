import os
import re
import tiktoken
import openai
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials as watsonxCredentials
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as TextParams
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as ChatParams
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv
from reactxen.experimental.wrapper.utils.prepare_chat_message import get_chat_message

load_dotenv()

completion_tokens = prompt_tokens = api_calls = 0
MAX_TOKENS = 4000

# Env variable for testing
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["WATSONX_APIKEY"] = os.getenv("WATSONX_APIKEY", "")
os.environ["WATSONX_URL"] = os.getenv("WATSONX_URL", "")
os.environ["AZURE_ENDPOINT"] = os.getenv("AZURE_ENDPOINT", "")
os.environ["AZURE_APIKEY"] = os.getenv("AZURE_APIKEY", "")
os.environ["API_VERSION"] = os.getenv("API_VERSION", "")

# This one we upper bound to control how much we should pass inside the tokenization
MAX_TEXT_LENGTH_TO_TOKENIZER = 3 * 1024 * 1024

# currently tested - different model needs difference reasoning
modelset = [
    "meta-llama/llama-3-70b-instruct",  # 0
    "ibm/granite-13b-chat-v2",  # 1
    "mistralai/mixtral-8x7b-instruct-v01",  # 2
    "ibm-meta/llama-2-70b-chat-q",  # 3
    "openai/gpt-3.5-turbo",  # 4
    "openai/gpt-4o",  # 5
    "mistralai/mistral-large",  # 6
    "meta-llama/llama-3-405b-instruct",  # 7
    # "ibm/granite-3-8b-instruct",  # 8
    "ibm/granite-4-h-small",  # 8
    "ibm/granite-3-3b-instruct",  # 9
    "meta-llama/llama-3-1-8b-instruct",  # 10
    "mistralai/mixtral-8x7b-instruct-v01",  # 11
    "meta-llama/llama-3-3-70b-instruct",  # 12
    "openai-azure/o1-preview",  # 13
    "ibm/granite-3-2-8b-instruct-preview-rc",  # 14
    "ibm/granite-3-2-8b-instruct",  # 15
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",  # 16
    "meta-llama/llama-4-scout-17b-16e-instruct",  # 17
    "openai-azure/gpt-4.1-2025-04-14",  # 18
    "ibm/granite-3-3-8b-instruct",  # 19
]


# this part is for get the context length
def get_context_length(model_id):
    context_dict = {
        "meta-llama/llama-3-70b-instruct": 8192,
        "ibm/granite-13b-chat-v2": 4096,
        "mistralai/mixtral-8x7b-instruct-v01": 16384,
        "ibm-meta/llama-2-70b-chat-q": 4096,
        "ibm/granite-4-h-small": 128000,
        "openai/gpt-3.5-turbo": 16384,  # Specific model added for clarity
        "openai/gpt-4o": 16384,  # Example context length for GPT-4
        "mistralai/mistral-large": 32768,
        "meta-llama/llama-3-405b-instruct": 16384,
        "ibm/granite-3-8b-instruct": 128000,
        "ibm/granite-3-3b-instruct": 8192,
        "meta-llama/llama-3-1-8b-instruct": 128000,
        "mistralai/mixtral-8x7b-instruct-v01": 8192,
        "meta-llama/llama-3-3-70b-instruct": 128000,
        "openai-azure/o1-preview": 128000,
        "ibm/granite-3-2-8b-instruct-preview-rc": 128000,
        "ibm/granite-3-2-8b-instruct": 128000,  # 15
        "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": 10000000,
        "meta-llama/llama-4-scout-17b-16e-instruct": 1000000,
        "openai-azure/gpt-4.1-2025-04-14": 10000000,  # 18
        "ibm/granite-3-3-8b-instruct": 128000,  # 19
    }

    if isinstance(model_id, str):
        if model_id in context_dict:
            return context_dict[model_id]
        else:
            raise ValueError(
                f"Invalid model_id: {model_id}. Not found in context_dict."
            )

    if 0 <= model_id < len(modelset):
        model_name = modelset[model_id]
        return context_dict.get(model_name, "Unknown model")
    else:
        raise IndexError(
            "Invalid model_id. Must be between 0 and {}".format(len(modelset) - 1)
        )


def trim_trailing_stop_sequence(generated_text: str, stop_sequences: list) -> str:
    for stop in stop_sequences:
        if generated_text.rstrip().endswith(stop):
            return generated_text.rstrip()[: -len(stop)].rstrip()
    return generated_text.rstrip()


def maybe_trim_generated_text(response: dict, stop_sequences: list) -> str:
    text = response.get("generated_text", "")
    if response.get("stop_reason") == "stop_sequence":
        return trim_trailing_stop_sequence(text, stop_sequences)
    return text.rstrip()


import requests

def direct_openai_llm(
    prompt,
    model_id="gpt-4o-mini",
    decoding_method="greedy",
    temperature=0.0,
    max_tokens=500,
    n=1,
    stop=None,
    seed=None,
    is_system_prompt=False,
):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    c_messages = get_chat_message(
        prompt, is_system_prompt=False, replace_system_by_assistant=True
    )
    if temperature == 0:
        temperature = 0.0
    request_params = {
        "model": model_id,
        "messages": c_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
    }
    if seed is not None:
        request_params["seed"] = seed
    try:
        response = client.chat.completions.create(**request_params)
    except Exception:
        # Gateway rejects logprobs: retry without it and let the caller see that
        # they are genuinely unavailable. Never substitute an estimate.
        request_params.pop("logprobs", None)
        response = client.chat.completions.create(**request_params)

    # OpenAI-compatible providers disagree on where logprobs live. OpenAI itself
    # returns logprobs.content[i].logprob; Together and several others return a
    # flat logprobs.token_logprobs list and leave .content as None. Reading only
    # the first shape yields None on the others -- silently, which is precisely
    # the failure this instrumentation exists to prevent. Try both.
    token_logprobs = None
    choice_logprobs = getattr(response.choices[0], "logprobs", None)
    if choice_logprobs is not None:
        content = getattr(choice_logprobs, "content", None)
        if content:
            token_logprobs = [t.logprob for t in content]
        else:
            flat = getattr(choice_logprobs, "token_logprobs", None)
            if flat:
                token_logprobs = [lp for lp in flat if lp is not None]

    generated_text = response.choices[0].message.content
    if stop:
        for phrase in stop:
            if phrase in generated_text:
                generated_text = generated_text.split(phrase)[0]
    response_object = {
        "generated_text": generated_text,
        "token_logprobs": token_logprobs,
        "logprobs_source": "provider" if token_logprobs else "unavailable",
        "promptTokens": response.usage.prompt_tokens,
        "input_token_count": response.usage.prompt_tokens,
        "completionTokens": response.usage.completion_tokens,
        "generated_token_count": response.usage.completion_tokens,
    }
    return response_object

def direct_gemini_llm(
    prompt,
    model_id="gemini-1.5-flash",
    decoding_method="greedy",
    temperature=0.0,
    max_tokens=500,
    n=1,
    stop=None,
    seed=None,
    is_system_prompt=False,
):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    if isinstance(prompt, list):
        text_prompt = ""
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_prompt += f"{role.upper()}: {content}\n"
    else:
        text_prompt = str(prompt)
    payload = {
        "contents": [{
            "parts": [{"text": text_prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    res_data = response.json()
    generated_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
    if stop:
        for phrase in stop:
            if phrase in generated_text:
                generated_text = generated_text.split(phrase)[0]
    usage = res_data.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", len(text_prompt) // 4)
    comp_tokens = usage.get("candidatesTokenCount", len(generated_text) // 4)
    response_object = {
        "generated_text": generated_text,
        "promptTokens": prompt_tokens,
        "input_token_count": prompt_tokens,
        "completionTokens": comp_tokens,
        "generated_token_count": comp_tokens,
    }
    return response_object

def watsonx_llm(
    prompt,
    model_id=8,
    decoding_method="greedy",
    temperature=0.0,
    max_tokens=500,
    n=1,
    stop=None,
    seed=None,
    is_system_prompt=False,
):
    # Get the model name from modelset
    if isinstance(model_id, str) and model_id in modelset:
        selected_model = model_id
    else:
        try:
            selected_model = modelset[model_id]
        except IndexError:
            raise ValueError(
                "Invalid model_id. Must be between 0 and {}".format(len(modelset) - 1)
            )

    if selected_model.startswith("openai-azure"):
        return azure_openai_llm(
            prompt=prompt,
            model_id=selected_model.replace("openai-azure/", ""),
            decoding_method=decoding_method,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            seed=seed,
        )

    if selected_model.startswith("openai/") or selected_model.startswith("openai-direct/"):
        # Strip only the leading routing prefix. A global replace corrupts model
        # ids that legitimately contain it: "openai/openai/gpt-oss-120b" (the
        # routing prefix plus Together's own id for that model) collapses to
        # "gpt-oss-120b", which the provider does not recognise.
        for _prefix in ("openai-direct/", "openai/"):
            if selected_model.startswith(_prefix):
                model_name = selected_model[len(_prefix):]
                break
        else:
            model_name = selected_model
        return direct_openai_llm(
            prompt=prompt,
            model_id=model_name,
            decoding_method=decoding_method,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            seed=seed,
            is_system_prompt=is_system_prompt,
        )

    if selected_model.startswith("google/") or selected_model.startswith("gemini/"):
        model_name = selected_model.replace("google/", "").replace("gemini/", "")
        return direct_gemini_llm(
            prompt=prompt,
            model_id=model_name,
            decoding_method=decoding_method,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            seed=seed,
            is_system_prompt=is_system_prompt,
        )

    if isinstance(stop, str):
        stop = [stop]

    parameters = {
        TextParams.RANDOM_SEED: seed,
        TextParams.TEMPERATURE: temperature,
        TextParams.MAX_NEW_TOKENS: max_tokens,
        TextParams.MIN_NEW_TOKENS: 1,
        TextParams.STOP_SEQUENCES: stop,
        # Ask the service for per-token logprobs. Without this the field is
        # absent, and callers previously filled the gap with a synthetic
        # generator instead of recording the absence.
        TextParams.RETURN_OPTIONS: {
            "token_logprobs": True,
            "generated_tokens": True,
        },
    }

    keys = os.environ.get("WATSONX_APIKEY", "")
    urls = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    credentials = watsonxCredentials(
        url=urls,
        api_key=keys,
    )

    # print("Here this i know its you....")
    model = ModelInference(
        model_id=selected_model,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
        max_retries=5,
        delay_time=2,
        retry_status_codes=[502, 503],
    )

    # Send the entire payload - ["generated_text"], promptTokens, completionTokens
    generated_response = None
    try:
        generated_response = model.generate(prompt=prompt)
    except Exception as e:
        print(f"Error occurred: {e}")
        # Some deployments reject RETURN_OPTIONS. Retry once without it so the
        # run continues, with logprobs honestly marked unavailable rather than
        # substituted. Also note the original swallowed this exception and then
        # dereferenced an unbound name; failures are now raised.
        retry_params = {
            k: v for k, v in parameters.items() if k != TextParams.RETURN_OPTIONS
        }
        try:
            model = ModelInference(
                model_id=selected_model,
                params=retry_params,
                credentials=credentials,
                project_id=project_id,
                max_retries=5,
                delay_time=2,
                retry_status_codes=[502, 503],
            )
            generated_response = model.generate(prompt=prompt)
        except Exception as retry_error:
            print(f"Retry without RETURN_OPTIONS also failed: {retry_error}")
            raise
    # print("Here this i know its completed....")

    result = generated_response["results"][0]

    # Normalise the logprob field so every provider path exposes the same key.
    token_logprobs = None
    generated_tokens = result.get("generated_tokens")
    if generated_tokens:
        collected = [
            t.get("logprob") for t in generated_tokens if t.get("logprob") is not None
        ]
        if collected:
            token_logprobs = collected
    result["token_logprobs"] = token_logprobs
    result["logprobs_source"] = "provider" if token_logprobs else "unavailable"

    return result
    # print (stop)
    # return_response = generated_response["results"][0]
    # return_response["generated_text"] = maybe_trim_generated_text(return_response, stop)
    # return return_response


def watsonx_llm_chat(
    prompt,
    model_id=8,
    decoding_method="greedy",
    temperature=0.0,
    max_tokens=500,
    n=1,
    stop=None,
    seed=None,
    is_system_prompt=False,
):
    keys = os.environ.get("WATSONX_APIKEY", "")
    urls = os.environ.get("WATSONX_URL", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")

    # ibm_watsonx_ai.metanames.GenChatParamsMetaNames().show()
    parameters = {
        # ChatParams.RANDOM_SEED: seed, # Not supported
        ChatParams.TEMPERATURE: temperature,
        ChatParams.MAX_TOKENS: max_tokens,
        # ChatParams.STOP_SEQUENCES: stop, # Not supported
    }

    credentials = watsonxCredentials(
        url=urls,
        api_key=keys,
    )

    # Get the model name from modelset
    if isinstance(model_id, str) and model_id in modelset:
        selected_model = model_id
    else:
        try:
            selected_model = modelset[model_id]
        except IndexError:
            raise ValueError(
                "Invalid model_id. Must be between 0 and {}".format(len(modelset) - 1)
            )

    replace_system_by_assistant = False
    if "mixtral-8x7B-instruct-v0.1".lower() in selected_model.lower():
        replace_system_by_assistant = True

    model = ModelInference(
        model_id=selected_model,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
        max_retries=5,
        delay_time=2,
        retry_status_codes=[502, 503],
        persistent_connection=False,
    )

    messages = get_chat_message(messages, is_system_prompt, replace_system_by_assistant)
    generated_response = model.chat(messages=prompt)
    return generated_response["choices"][0]


def azure_openai_llm(
    prompt,
    model_id="o1-preview",
    decoding_method="greedy",
    temperature=0.0,
    max_tokens=500,
    n=1,
    stop=None,
    seed=None,
    is_system_prompt=False,
):
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        api_key=os.environ["AZURE_APIKEY"],
        api_version=os.environ["API_VERSION"],
    )

    # print (stop)

    if stop:
        if "\nAction" not in stop and any("Observation" in item for item in stop):
            # this is for ThinkandActTogether
            prompt += "\nNow, only generate Thought, Action and Action Input one time based on recent observation."
        elif any("Thought" in item for item in stop):
            prompt += (
                "\nNow, only generate Action and Action Input based on recent thought."
            )
        elif any("Action" in item for item in stop):
            prompt += "\nNow, only generate Thought."
        else:
            pass
    else:
        pass
        # print ('you came here')

    c_messages = get_chat_message(
        prompt, is_system_prompt=False, replace_system_by_assistant=True
    )

    max_tokens = 32000

    if temperature == 0:
        temperature = 1.0
        seed = 20

    request_params = {
        "model": model_id,  # Use direct model ID without modifications
        "messages": c_messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }

    # Only include `stop` if it's not None
    # if stop is not None:
    #    request_params["stop"] = stop

    response = client.chat.completions.create(**request_params)
    stop_phrases = stop  # Define your own stop markers
    generated_text = response.choices[0].message.content

    if stop:
        for phrase in stop_phrases:
            if phrase in generated_text:
                generated_text = generated_text.split(phrase)[0]

    response_object = {}
    response_object["generated_text"] = generated_text
    response_object["promptTokens"] = response.usage.prompt_tokens
    response_object["input_token_count"] = response.usage.prompt_tokens
    response_object["completionTokens"] = response.usage.completion_tokens
    response_object["generated_token_count"] = response.usage.completion_tokens

    # print(response)
    return response_object


def openaicall(messages, n=1, temperature=0.0, max_tokens=150, stop=None):
    # Set default for 'stop' if not provided
    if stop is None:
        stop = []

    # Ensure 'stop' is a list if a string is provided
    if isinstance(stop, str):
        stop = [stop]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Handle multiple responses if n > 1
    responses = []
    completion_tokens = 0
    prompt_tokens = 0
    api_calls = 0

    for _ in range(n):
        chat_completion = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": messages}]
        )

        completion_text = chat_completion.choices[0].message.content.strip()
        responses.append(completion_text)

        completion_tokens = chat_completion.usage.completion_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens

        # Update token counts
        completion_tokens += chat_completion.usage.completion_tokens
        prompt_tokens += chat_completion.usage.prompt_tokens
        api_calls += 1

    return responses if n > 1 else responses[0]  # Return list or single response


# def return_usage():
#     """
#     Returns the usage statistics for API calls, including
#     completion tokens, prompt tokens, and the number of API calls made.

#     Returns:
#         tuple: A tuple containing completion tokens, prompt tokens, and API calls.
#     """
#     global completion_tokens, prompt_tokens, api_calls
#     return completion_tokens, prompt_tokens, api_calls


def gpt_usage(backend="bam"):
    """
    Calculate the usage and cost of API calls based on token counts.

    Args:
        backend (str): The backend model used (default is "bam").

    Returns:
        dict: A dictionary containing completion tokens, prompt tokens, and the calculated cost.
    """
    global completion_tokens, prompt_tokens

    # Define pricing based on the backend
    pricing = {
        "gpt-4": {
            "completion": 0.06,
            "prompt": 0.03,
        },
        "bam": {
            "completion": 0.002,
            "prompt": 0.0015,
        },
    }

    if backend not in pricing:
        raise ValueError(
            f"Unknown backend '{backend}'. Valid options are: {list(pricing.keys())}"
        )

    # Calculate cost based on the selected backend
    cost = (
        completion_tokens / 1000 * pricing[backend]["completion"]
        + prompt_tokens / 1000 * pricing[backend]["prompt"]
    )

    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }


def count_tokens(
    input_text, model_id=None, upper_limit=10000000, skip_token_counting=False
):
    """
    Counts the number of tokens in the given input based on the specified model.

    Args:
        input_text (str): The input text for which to count tokens.
        model_id (str): The model ID to use for tokenization. Defaults to None for selected model.

    Returns:
        int: The count of tokens in the input text.
    """

    # avoid counting the token
    if skip_token_counting:
        return 1

    if isinstance(model_id, str) and model_id in modelset:
        selected_model = model_id
    else:
        selected_model = modelset[model_id]  # Ensure you have access to selected_model

    # Handle OpenAI model token counting
    if "openai" in selected_model:
        return openai_count_tokens(
            input_text, selected_model
        )  # Adjust this function call if necessary

    # print (upper_limit)
    return watsonx_count_tokens(input_text, selected_model, upper_limit)


def watsonx_count_tokens(text, model_id="mistralai/mistral-large", upper_limit=1000000):
    credentials = watsonxCredentials(
        url=os.environ["WATSONX_URL"],
        api_key=os.environ["WATSONX_APIKEY"],
    )

    # print ('started -----------')
    # print (upper_limit)
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=os.environ["WATSONX_PROJECT_ID"],
        max_retries=2,
        delay_time=2,
        retry_status_codes=[502, 503],
        # persistent_connection=False,    # this did not work with tokenize
    )

    if len(text) > MAX_TEXT_LENGTH_TO_TOKENIZER:
        text = text[0:MAX_TEXT_LENGTH_TO_TOKENIZER]

    total_count = upper_limit
    # print(upper_limit)
    try:
        tokenized_response = model.tokenize(prompt=text, return_tokens=False)
        total_count = tokenized_response["result"]["token_count"]
        # print(total_count)
        # print("-------------")
    except ApiRequestFailure as error_message:
        match = re.search(
            r'"the number of input tokens (\d+) cannot exceed"', error_message
        )
        if match:
            total_count = match.group(1)
            # print("Extracted token count:", token_count)
        else:
            total_count = upper_limit + 10
    except Exception as ex:
        pass
    # print ('ended -----------')
    return total_count


def openai_count_tokens(text, model="o1-preview", is_chat=False):
    total_token = None
    openai_model = False
    if model.startswith("openai-azure/"):
        openai_model = True
        model = model.replace("openai-azure/", "")
    try:
        try:
            enc = tiktoken.encoding_for_model(model)
        except:
            if openai_model:
                # print(openai_model)
                enc = tiktoken.get_encoding("o200k_base")
            else:
                raise KeyError(f"Could not find encoding for {model}.")
        if not is_chat:
            tokens = enc.encode(text)
            total_token = len(tokens)
        else:
            total_token = 0
            for message in text:
                total_token += 4  # Base metadata overhead
                total_token += len(enc.encode(message))
            total_token += 2  # Assistant reply overhead
    except KeyError:
        print(f"Could not find encoding for {model}.")

    return total_token
