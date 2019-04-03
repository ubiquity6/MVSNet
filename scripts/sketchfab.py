import os
import json
import requests

##
# Uploading a model to Sketchfab. You will need to supply an API token for upload to work
#
# 1. Upload a model. If the upload is successful, the API will return
#    the model's future URL, and the model will be placed in the processing queue
#

SKETCHFAB_DOMAIN = 'sketchfab.com'
SKETCHFAB_API_URL = 'https://api.{}/v3'.format(SKETCHFAB_DOMAIN)


def _get_request_payload(api_token, data={}, files={}, json_payload=False):
    """Helper method that returns the authentication token and proper content
    type depending on whether or not we use JSON payload."""
    headers = {'Authorization': 'Token {}'.format(api_token)}

    if json_payload:
        headers.update({'Content-Type': 'application/json'})
        data = json.dumps(data)

    return {'data': data, 'files': files, 'headers': headers}


def upload(model_file, api_token='5f2cad0a1eff43699ebe0765b8d9a8ba', name='', description=''):
    """POST a model to sketchfab.
    This endpoint only accepts formData as we upload a file. You will 
    need a Sketchfab API token. See:
    https://help.sketchfab.com/hc/en-us/articles/202600683-Finding-your-API-Token
    """
    model_endpoint = os.path.join(SKETCHFAB_API_URL, 'models')

   # password = 'my-password'  # requires a pro account
   # private = 1  # requires a pro account
    tags = ['bob', 'character', 'video-games']  # Array of tags
    categories = ['people']  # Array of categories slugs
    license = 'CC Attribution'  # License label
    isPublished = True,  # Model will be on draft instead of published
    isInspectable = True,  # Allow 2D view in model inspector

    data = {
        'name': name,
        'description': description,
        'tags': tags,
        'categories': categories,
        'license': license,
        # 'private': private,
        # 'password': password,
        'isPublished': isPublished,
        'isInspectable': isInspectable
    }

    f = open(model_file, 'rb')

    files = {'modelFile': f}

    print('Uploading ...')

    try:
        r = requests.post(
            model_endpoint, **_get_request_payload(api_token,
                                                   data, files=files))
    except requests.exceptions.RequestException as e:
        print('An error occured: {}'.format(e))
        return
    finally:
        f.close()

    if r.status_code != requests.codes.created:
        print('Upload failed with error: {}'.format(r.json()))
        return

    uid = r.json()['uid']
    model_url = 'https://sketchfab.com/models/{}'.format(uid)
    print('Upload successful. Your model is being processed.')
    print('Once the processing is done, the model will be available at: {}'.format(
        model_url))

    return model_url
