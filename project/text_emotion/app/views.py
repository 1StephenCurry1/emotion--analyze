from django.core.paginator import Paginator
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from user.models import UserTable
from .models import Text
from .lstm.serve import predict_main


def login(req):
    return render(req, 'login.html')


def register(req):
    return render(req, 'register.html')


def index(req):
    username = req.session['username']
    return render(req, 'index.html', locals())


def login_out(req):
    del req.session['username']
    return HttpResponseRedirect('/')


def personal(req):
    username = req.session['username']
    role_id = req.session['role']
    user = UserTable.objects.filter(name=username).first()
    return render(req, 'personal.html', locals())


def get_text(request):
    """
    获取用户列表信息 | 模糊查询
    :param request:
    :return:
    """
    keyword = request.GET.get('name')
    page = request.GET.get("page", '')
    limit = request.GET.get("limit", '')
    role_id = request.GET.get('position', '')
    response_data = {}
    response_data['code'] = 0
    response_data['msg'] = ''
    data = []
    if keyword is None:
        results = Text.objects.all()
        paginator = Paginator(results, limit)
        results = paginator.page(page)
        if results:
            for user in results:
                record = {
                    "id": user.id,
                    "title": user.title,
                    "type": user.type,
                    "owner": user.owner,
                    'create_time': user.create_time.strftime('%Y-%m-%d %H:%m:%S'),
                }
                data.append(record)
            response_data['count'] = len(Text.objects.all())
            response_data['data'] = data
    else:
        users_all = Text.objects.filter(name__contains=keyword).all()
        paginator = Paginator(users_all, limit)
        results = paginator.page(page)
        if results:
            for user in results:
                record = {
                    "id": user.id,
                    "title": user.title,
                    "type": user.type,
                    "owner": user.owner,
                    'create_time': user.create_time.strftime('%Y-%m-%d %H:%m:%S'),
                }
                data.append(record)
            response_data['count'] = len(users_all)
            response_data['data'] = data
    return JsonResponse(response_data)


def text_manage(req):
    username = req.session['username']
    role_id = req.session['role']
    user = UserTable.objects.filter(name=username).first()
    return render(req, 'text_manage.html', locals())


def del_text(request):
    """
    删除用户
    """
    user_id = request.POST.get('id')
    result = Text.objects.filter(id=user_id).first()
    try:
        if not result:
            response_data = {'error': '删除信息失败！', 'message': '找不到id为%s的文本' % user_id}
            return JsonResponse(response_data, status=403)
        result.delete()
        response_data = {'message': '删除成功！'}
        return JsonResponse(response_data, status=201)
    except Exception as e:
        response_data = {'message': '删除失败！'}
        return JsonResponse(response_data, status=403)


def predict(req):
    text = req.POST.get('text')
    print(text)
    try:
        result = predict_main(text)
        if result == 'NEG':
            result = '消极'
        else:
            result = '积极'
        Text.objects.create(
            title=text,
            type=result,
            owner=req.session['username'],

        )
        response_data = {'result': result}
        return JsonResponse(response_data, status=201)
    except Exception as e:
        response_data = {'msg': '分类失败'}

        return JsonResponse(response_data, status=505)

def text_classify(req):
    username = req.session['username']
    role_id = req.session['role']
    user = UserTable.objects.filter(name=username).first()
    return render(req, 'text_classify.html', locals())