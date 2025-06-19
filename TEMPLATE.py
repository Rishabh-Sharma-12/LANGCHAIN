from langchain_core import prompts,prompt_template

template=prompt_template(
    template="""
        Helloooo
    """
    ,
    input_variables(['paper','style','lengths'])
)
template.Save(template.json)
