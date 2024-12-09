# imports - clean
import streamlit as st

from clarifai.client.app import App
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.client.workflow import Workflow
from clarifai.modules.css import ClarifaiStreamlitCSS

from clarifai_grpc.grpc.api import (
    resources_pb2,
    service_pb2,
)  # hopefully temporary for the Search component
from annotated_text import annotated_text  # https://github.com/tvst/st-annotated-text
import ast


# streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


# This must be within the display() function
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

##########################
#### HELPER FUNCTIONS ####
##########################

def footer(st):
    with open('footer.html', 'r') as file:
        footer = file.read()
        st.write(footer, unsafe_allow_html=True)


promptLLM = """I need your help with image moderation. Please assess the following image for content that falls into any of the following categories: nudity, drugs, violence, gore, and other NSFW (Not Safe For Work) content. Ignore any text overlays, such as 'Image blocked on upload by AI' or existing NSFW scores, as these may be erroneous. For each category, provide a clear evaluation and specify any detected elements that may be of concern.

Assign an NSFW score between 0-100 based on the following categories:

1. Score 0: Safe (0) - The image is entirely safe and appropriate for all audiences. No nudity, violence, drug use, suggestive content, or any other disturbing elements.

2. Score 20: Mildly Suggestive (20) - The image contains mildly suggestive elements that are not explicit but may be slightly provocative or flirtatious.

3. Score 40: Moderately Inappropriate (40) - The image includes moderately inappropriate content, such as revealing clothing, mild violence, or suggestive poses.

4. Score 60: Explicit Content (60) - The image contains explicit content like partial nudity, visible drug paraphernalia, or moderate violence and gore.

5. Score 80: Highly Inappropriate (80) - The image includes highly inappropriate content such as full nudity, significant violence, weapons, guns, knives, or visible drug use.

6. Score 100: Extreme NSFW (100) - The image contains extremely explicit or graphic content, such as pornography, severe violence, or extreme gore.

The goal is to ensure the image is safe for professional, public, and family-friendly environments. Clearly indicate whether the image should be flagged or passed based on the evaluation criteria and the assigned NSFW score, without considering erroneous text. Here is an example output:

"Summary: The image depicts a person with what appears to be blood on their arm and mouth, which could be interpreted as self-harm or violence. Based on the content, here is the evaluation:

Nudity: There is no nudity in the image.

Drugs: There are no visible drugs or drug paraphernalia.

Violence: The image suggests violence or self-harm due to the presence of blood and the way it is depicted.

Weapon: There is no visible weapon, knife, or gun in the image.

Gore: The image contains blood, which can be considered gory.

Other NSFW Content: The image is disturbing due to the implication of self-harm or violence.

NSFW Score: 80 (Highly Inappropriate)

Recommendation: The image should be flagged due to the depiction of blood and the implication of self-harm or violence, making it highly inappropriate for professional, public, and family-friendly environments."
format output as a json-formatted
"""


business_prompt = promptLLM

####################
####  SIDEBAR   ####
####################

with st.sidebar:

    #### Page Display Options

    st.subheader("Page Setup")

    page_title = st.text_input(label="Module Title", value="Content Moderation Demo")

    st.divider()

    company_logo = st.text_input(
        label="Banner Url",
        value="https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png",
    )
    company_logo_width = st.slider(
        label="Banner Width", min_value=1, max_value=1000, value=300
    )

    st.divider()

    #### Image URLs
    st.subheader("Image URLs")

    image_urls = st.text_area(height = 300,
            label = 'Image URLs',
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/moderation_1.png'
        )

    #### Model Selections
    st.divider()

    st.subheader("Model Selections")

    # Visual Classifier Selection

    community_visual_classifiers = list(
        App(pat=auth._pat).list_models(
            filter_by={"model_type_id": "visual-classifier"}, only_in_app=False
        )
    )
    community_visual_classifiers_ids_only = [x.id for x in community_visual_classifiers]

    print(
        f"community_visual_classifiers_ids_only: {community_visual_classifiers_ids_only}"
    )

    vis_class_model_id = st.selectbox(
        label="Select Image Classification",
        options=community_visual_classifiers_ids_only,
        index=19,  # this should hopefully be the `moderation-all-resnext-2` model selected
    )

    selected_vis_clas_model = [
        x for x in community_visual_classifiers if x.id == vis_class_model_id
    ][0]
    vis_class_model_name = selected_vis_clas_model.name
    vis_class_user_id = selected_vis_clas_model.user_id
    vis_class_app_id = selected_vis_clas_model.app_id

    vis_class_max_concepts = st.slider(
        label="Specify max concepts", min_value=1, max_value=200, value=20
    )

    vis_class_threshold = st.slider(
        label="Specify threshold for Visual Classifier Models", min_value=0.0, max_value=1.0, value=0.5
    )

    workflow_threshold = st.slider(
        label="Specify threshold for Moderation Workflow", min_value=0.0, max_value=1.0, value=0.5
    )

    # LLVM Selection
    st.write("")
    st.write("")

    community_llvms = list(
        App(pat=auth._pat).list_models(
            filter_by={"model_type_id": "multimodal-to-text"}, only_in_app=False
        )
    )
    community_llvms_ids_only = [x.id for x in community_llvms]

    llvm_model_id = st.selectbox(
        label="Select LLVM",
        options=community_llvms_ids_only,
        index=11,  # this should hopefully be the `gpt-4o`
    )

    selected_llvm = [x for x in community_llvms if x.id == llvm_model_id][0]
    llvm_model_name = selected_llvm.name
    llvm_user_id = selected_llvm.user_id
    llvm_app_id = selected_llvm.app_id

    # llvm inference params
    llvm_temp = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.3)

    llvm_max_tokens = st.number_input(label="Max Tokens", value=1024)

    llvm_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=0.8)

    technical_prompt = st.text_area(
        label="Technical Prompt",
        value="Only response to the following prompt in a json-formatted list of 30 individual concepts/classes.",
    )

    #### Output Display options
    st.divider()

    st.subheader("Output Display Options")

    tag_bg_color = st.color_picker(label="Tag Background Color (hex)", value="#aabbcc")

    tag_text_color = st.color_picker(label="Tag Text Color (hex)", value="#2B2D37")


####################
####  MAIN PAGE ####
####################

st.image(company_logo, width=company_logo_width)

st.title(page_title)


tab1, tab2 = st.tabs(["Upload by URL", "Upload by file"])

# image url version
with tab1:
    with st.form(key="input-data-url"):

        # url version
        upload_image_url = st.text_input(
            label="Enter image url:",
            value="https://samples.clarifai.com/metro-north.jpg",
        )

        business_prompt_url = st.text_area(
            label="Enter prompt for Large Language Visual Model (LLVM)",
            value=business_prompt,
            height=100,
        )

        submitted_url = st.form_submit_button("Upload")

# local file version
with tab2:
    with st.form(key="input-data-file"):
        upload_image_file = st.file_uploader(
            label="Upload image file", type=["jpg", "jpeg", "png", "gif"]
        )

        business_prompt_file = st.text_area(
            label="Enter prompt for Large Language Visual Model (LLVM)",
            value=business_prompt,
            height=100,
        )

        submitted_file = st.form_submit_button("Upload")


if submitted_url or submitted_file:

    #### display image

    # figure out which one was submitted
    if submitted_url == True:
        st.image(upload_image_url, caption=upload_image_url)
        business_prompt = business_prompt_url
    if submitted_file == True:
        st.image(upload_image_file, caption="Uploaded Image File")
        image_bytes = upload_image_file.read()
        business_prompt = business_prompt_file

    #### Visual Classifier Output

    st.subheader(f"Visual Classifier Output(s):")
    with st.spinner():
        # with st.expander(label=f'{vis_class_model_name}'):
        st.write(f"{vis_class_model_name} | {vis_class_model_id}")

        vis_class_model = Model(
            pat=auth._pat,
            model_id=vis_class_model_id,
            user_id=vis_class_user_id,
            app_id=vis_class_app_id,
        )

        if submitted_url == True:
            vis_class_pred = vis_class_model.predict_by_url(
                url=upload_image_url,
                input_type="image",
                inference_params={"max_concepts": vis_class_max_concepts},
            )

        if submitted_file == True:
            vis_class_pred = vis_class_model.predict_by_bytes(
                input_bytes=image_bytes,
                input_type="image",
                inference_params={"max_concepts": vis_class_max_concepts},
            )

        vis_class_threshold = vis_class_threshold
        filtered_concepts = [
            x
            for x in vis_class_pred.outputs[0].data.concepts
            if x.value >= vis_class_threshold
        ]
        if not filtered_concepts:
            st.info(f"No concepts detected above the confidence threshold of {vis_class_threshold:.2f}")
        
        else:
            vis_class_tuple_of_tuples = tuple(
            [
                (f"{x.name}", f"{x.value:.3f}", tag_bg_color, tag_text_color)
                for x in filtered_concepts
            ]
        )

            list_with_empty_strings = []
            for item in vis_class_tuple_of_tuples:
                list_with_empty_strings.append(item)
                list_with_empty_strings.append(" ")  # Add an empty string after each item

            # Remove the last empty string as it's not needed
            if list_with_empty_strings[-1] == "":
                list_with_empty_strings.pop()

            # Convert back to a tuple if needed
            vis_class_tuple_of_tuples = tuple(list_with_empty_strings)

            annotated_text(*vis_class_tuple_of_tuples)


    #### Workflow Output

    st.subheader(f"Workflow Output(s):")
    with st.spinner():
        workflow_url = "https://clarifai.com/clarifai/Momio-Debug/workflows/Momio-Image-moderation-3"
        # Creating the workflow
        vis_class_workflow = Workflow(
            url=workflow_url,
            pat=auth._pat,
        )

        if submitted_url == True:
            vis_class_pred2 = vis_class_workflow.predict_by_url(
                url=upload_image_url,
                input_type="image",
            )

        if submitted_file == True:
            vis_class_pred2 = vis_class_workflow.predict_by_bytes(
                input_bytes=image_bytes,
                input_type="image",
            )

        print(vis_class_pred2.results[0].outputs)
        vis_class_tuple_of_tuples2 = ()

        for output2 in vis_class_pred2.results[0].outputs[:]:
            print(f"model_id: {output2.model.id} **************")
            print(f"data: *****\n{output2.data}")

            if (
                output2.model.id == "moderation-all-resnext-2"
                or output2.model.id == "nsfw-recognition"
                or output2.model.id == "moderation-recognition"
                or output2.model.id == "moderation-multilingual-text-classification"
                or output2.model.id == "weapon-detection"
            ):

                workflow_threshold = workflow_threshold

                filtered_concepts = [
                    x
                    for x in output2.data.concepts
                    if x.value >= workflow_threshold
                ]
                if not filtered_concepts:
                    st.info(f"No concepts detected above the confidence threshold of {workflow_threshold:.2f}")
                else:
                    vis_class_tuple_of_tuples2 = vis_class_tuple_of_tuples2 + tuple(
                        [
                            (f"{x.name}", f"{x.value:.3f}", tag_bg_color, tag_text_color)
                            for x in filtered_concepts
                        ]
                    )

            if output2.model.id == "weapon-detection" and hasattr(
                output2.data, "regions"
            ):
                weapon_regions = output2.data.regions
                highest_value = 0.00
                # loop through the regions and find the highest value of prediction
                for weapon_region in weapon_regions:
                    print(f"Weapon Data: {weapon_region} ")
                    current_value = float(weapon_region.value)
                    if current_value >= highest_value:
                        highest_value = current_value

                if highest_value > 0.5:
                    vis_class_tuple_of_tuples2 = vis_class_tuple_of_tuples2 + tuple(
                        [
                            (
                                f"Weapon",
                                f"{highest_value:.3f}",
                                tag_bg_color,
                                tag_text_color,
                            )
                        ]
                    )

        print(f"tuples: {vis_class_tuple_of_tuples2}")

        annotated_text(*vis_class_tuple_of_tuples2)

    ### LLVM Output
    st.subheader(f"LLVM Output(s):")
    with st.spinner():
        llvm_class_model = Model(
            pat=auth._pat,
            model_id=llvm_model_id,
            user_id=llvm_user_id,
            app_id=llvm_app_id,
        )

        llvm_inference_params = {
            "temperature": llvm_temp,
            "max_tokens": llvm_max_tokens,
            "top_p": llvm_top_p,
        }

        if submitted_url == True:
            llvm_pred = llvm_class_model.predict(
                inputs=[
                    Inputs.get_multimodal_input(
                        input_id="",
                        image_url=upload_image_url,
                        raw_text=f"{business_prompt}",
                    )
                ],
                inference_params=llvm_inference_params,
            )

        if submitted_file == True:
            llvm_pred = llvm_class_model.predict(
                inputs=[
                    Inputs.get_multimodal_input(
                        input_id="",
                        image_bytes=image_bytes,
                        raw_text=f"{business_prompt}",
                    )
                ],
                inference_params=llvm_inference_params,
            )

        # wrangling output and cleaning up / converting to labels
        llvm_output = llvm_pred.outputs[0].data.text.raw

        print(f"llvm_output:\n{llvm_output}")
        st.write(llvm_output)

# Collapsible text field at the bottom

with st.expander("Details"):
    st.markdown("""
    ### About Content Moderation
    Content moderation uses AI models to automatically analyze images and identify potentially inappropriate, unsafe, or harmful content. This helps ensure image quality, safety, and compliance across different platforms and use cases.
    
    ### How This Works
    1. **Image Upload**:
        - Upload an image via URL or local file.
        - Supported formats: JPEG, JPG, PNG, GIF.
    
    2. **Model Analysis**:
        - Click "Upload" to analyze the image.
        - Three detailed outputs will be generated:
            1. **Visual Classification**: Identifies and categorizes image contents using models from the Clarifai community and custom-trained models. Some examples include:
                - general-image-recognition
                - moderation-recognition
                - nsfw-recognition  
                Select the model from the sidebar. These are highly efficient and fast.
            2. **Workflow Analysis**: Uses this [Image-moderation](https://clarifai.com/clarifai/Momio-Debug/workflows/Momio-Image-moderation-3) workflow, which contains various models such as:
                - **moderation-all-resnext-2**: Detects overall inappropriate or offensive content.
                - **nsfw-recognition**: Identifies not-safe-for-work (NSFW) elements like nudity or adult themes.
                - **moderation-recognition**: Performs additional content moderation checks.
                - **moderation-multilingual-text-classification**: Analyzes any textual elements within the image.
                - **weapon-detection**: Looks for the presence of weapons or other security threats.
            3. **LLVM (Large Language Visual Model)**: Provides a detailed, contextual description of the image based on your prompt. Examples include:
                - GPT-4o
                - Claude-3-Vision  
                Select different available models from the sidebar.
    
    3. **Customization Options from the Sidebar**:
        - Select different visual classification models.
        - Choose various Large Language Vision Models.
        - Adjust model parameters like temperature and max tokens.
        - Customize output display colors.
    """)

    # Text area for additional notes
    project_details = st.text_area(
        "Add Your Notes:",
        height=100,
        placeholder="Add any additional notes..."
    )
    if project_details:
        st.markdown("### Your Notes:")
        st.write(project_details)

####################
####  FOOTER    ####
####################

footer(st)