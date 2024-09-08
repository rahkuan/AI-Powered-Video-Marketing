from langchain_openai import OpenAI

llm = OpenAI()

def build_context_data(product_name='', person_description=''):
    context = f"I have a {product_name} and want to advertising it to a person with description as [{person_description}]."
    return context

def get_personalized_voiceover_text(context):
    text = f'{context} I need a voiceover content for the marketing video roughly 10 seconds in timespan targeted to that person. Your answer is:'
    response = llm.invoke(text)
    return response

def get_personalized_music_type(context):
    text = f'{context} I need a music type like classical, jazz, ambient, rock, electronic, etc,... for an background music in an marketing video targeted to that person. My question is: what is your recommendation in few word?'
    response = llm.invoke(text)
    
    if len(response) > 20:
        text = f'given your recommendation of music type from following:\n{response}\n Please point out music type you has recommended'
        response = llm.invoke(text)
        return response

if __name__ == '__main__':
    #product_name = 'watch'
    #person_description = 'living in Asia, female, age 40'
    #context = build_context_data(product_name, person_description)
    #print (context)

    # Personalized voiceover content
    #print ('Personalized voiceover content')
    #personalized_voiceover_text = get_personalized_voiceover_text(context)
    #print (personalized_voiceover_text)

    # Personalized music type
    #print ('Personalized music background')
    #personalized_music_style = get_personalized_music_type(context)
    #print (personalized_music_style)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_description", type=str, help="description in text of the product, at first it is product name, for example: watch")
    parser.add_argument("--person_description", type=str, help="keywords seperated by comma contain information about a person that video ad is target to, for example: female, living in Asia, age 40")
    args = parser.parse_args()
    context = build_context_data(args.product_description, args.person_description)
    personalized_voiceover_text = get_personalized_voiceover_text(context)
    print ('personalized_voiceover_text')
    print (personalized_voiceover_text)

    personalized_music_style = get_personalized_music_type(context)
    print ('personalized_music_style')
    print (personalized_music_style)        
