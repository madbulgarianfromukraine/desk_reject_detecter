import json
import requests
from openreview import OpenReviewException
import openreview.api

def format_params(params):
    if isinstance(params, dict):
        formatted_params = {}
        for key, value in params.items():
            formatted_params[key] = format_params(value)
        return formatted_params

    if isinstance(params, list):
        formatted_params = []
        for value in params:
            formatted_params.append(format_params(value))
        return formatted_params

    if isinstance(params, bool):
        return json.dumps(params)

    return params

def __handle_response(response):
    try:
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if 'application/json' in response.headers.get('Content-Type', ''):
            error = response.json()
        elif response.text:
            error = {
                'name': 'Error',
                'message': response.text
            }
        else:
            error = {
                'name': 'Error',
                'message': response.reason
            }
        raise OpenReviewException(error)

EDITS_URL = "https://openreview.net/notes/edits/"


def get_attachment(client: openreview.api.OpenReviewClient, field_name, id=None, ids=None, group_id=None, invitation_id=None):
    """
    Gets the binary content of a attachment using the provided note id
    If the pdf is not found then this returns an error message with "status":404.

    :param field_name: name of the field associated with the attachment file
    :type field_name: str
    :param id: Note id or Reference id of the pdf
    :type id: str
    :param ids: List of Note ids or Reference ids. The max number of ids is 50
    :type id: list[str]
    :param group_id: Id of group where attachment is stored
    :type group_id: str
    :param invitation_id: Id of invitation where attachment is stored
    :type invitation_id: str

    :return: The binary content of a pdf
    :rtype: bytes

    Example:

    >>> f = get_attachment(id='Place Note-ID here', field_name='pdf')
    >>> with open('output.pdf','wb') as op: op.write(f)

    """

    if not any([id, ids, group_id, invitation_id]):
        raise OpenReviewException('Provide exactly one of the following: id, ids, group_id, invitation_id')

    params = {}
    params['name'] = field_name

    if id:
        url = client.baseurl
        params['id'] = id
    elif ids:
        url = client.baseurl
        params['ids'] = ','.join(ids)
    elif group_id:
        url = client.groups_url
        params['id'] = group_id
    elif invitation_id:
        url = client.invitations_url
        params['id'] = invitation_id

    response = client.session.get(EDITS_URL + '/attachment', params=format_params(params), headers=client.headers)
    response = __handle_response(response)

    return response.content