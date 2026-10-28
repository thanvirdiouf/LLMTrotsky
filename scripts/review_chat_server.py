"""Local-only review workbench: verified samples, source lookup and optional live endpoint."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,secrets,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from http.cookies import SimpleCookie,CookieError
from pathlib import Path
from urllib.parse import urlsplit
from grounded_chat import ROOT,MODEL
from grounded_request_boundary import ConversationState,apply_event
from review_chat import load_snapshot,prepare_turn,parse_reply,append_turn

STATIC=ROOT/'web/review_chat'

class Workbench:
    def __init__(self,snapshot,review_run,feedback_path,*,base_url=None,model=MODEL['model_id'],tokenizer=None):
        self.profile,self.collection,self.ledger=load_snapshot(snapshot)
        self.review_run=Path(review_run);self.feedback_path=Path(feedback_path)
        self.base_url=base_url;self.model=model;self.tokenizer=tokenizer
        self.sessions={};self.lock=threading.Lock()
    def bootstrap(self):
        return {'title':'Trotsky, in the present','fictional_identity':True,'source_snapshot_date':self.collection['reference_date'],
            'sources':[{k:p[k] for k in ('id','title','author','domain','composition_date_interval')} for p in self.collection['packets']],
            'live_available':bool(self.base_url),'live_model_requested':self.model,
            'live_destination':urlsplit(self.base_url).netloc if self.base_url else None,
            'model_identity':'endpoint operator asserted; not remotely attested' if self.base_url else None,
            'samples_available':bool(self.review_rows()),'feedback_local_only':True}
    def review_rows(self):
        report_path=self.review_run/'verification.json'
        if not report_path.is_file():return []
        report=json.loads(report_path.read_text())
        if report.get('status')!='verified' or report.get('frozen_pipeline_replayed') is not True:return []
        answers=self.review_run/'download/ghost_review_chat_results/answers.jsonl';raw=answers.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=report['answers_sha256']:raise ValueError('Verified review answers changed')
        rows=[]
        for line in raw.decode().splitlines():
            r=json.loads(line)
            # Raw model outputs/reasoning and full request prompts are not served.
            rows.append({k:r[k] for k in ('case_id','title','turn','question','status','reply','validation_error','model','human_review_status')}
                |{'source_packets':r['request']['source_packets'],'retrieval':r['request']['retrieval'],
                  'omitted_history_pairs':r['request']['omitted_history_pairs'],
                  'token_budget_omitted_packets':r['request']['token_budget_omitted_packets']})
        return rows
    def session(self,identity):
        with self.lock:
            if identity not in self.sessions:
                self.sessions[identity]={'state':ConversationState(tuple(p['id'] for p in self.collection['packets'])),'busy':threading.Lock()}
            return self.sessions[identity]
    def ask(self,identity,body,live=False):
        if set(body)-{'question','source_ids','detail','reflection_allowed'} or not isinstance(body.get('question'),str) or len(body['question'])>12000:
            raise ValueError('Supply a question within12000characters and supported controls')
        session=self.session(identity)
        if not session['busy'].acquire(blocking=False):raise ValueError('A request is already running in this conversation')
        try:
            state=session['state'];inventory=[p['id'] for p in self.collection['packets']]
            state,_=apply_event(state,{'event':'set_source_scope','source_ids':body.get('source_ids',list(state.source_scope))},inventory)
            state,_=apply_event(state,{'event':'set_detail','detail':body.get('detail',state.detail)},inventory)
            if live and not self.base_url:raise ValueError('Live inference is not configured. Source lookup makes no model call.')
            if live and len(state.prior_turns)>=20:raise ValueError('Review limit reached: start a new conversation after ten model replies')
            request=prepare_turn(self.profile,self.collection,self.ledger,state,body['question'],tokenizer=self.tokenizer if live else None,
                reflection_allowed=body.get('reflection_allowed',True),today=datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat())
            if not live:
                return {'kind':'source_lookup','message':'Candidate passages for inspection; this is not a model answer.',
                    'source_packets':request['source_packets'],'retrieval':request['retrieval'],'model_calls':0}
            import requests
            from remote_model import call_chat
            with requests.Session() as http:
                text,finish,usage=call_chat(http,self.base_url,{'model':self.model,'messages':request['messages'],'temperature':0,
                    'max_tokens':1024,'stream':False,'chat_template_kwargs':{'enable_thinking':False}},os.getenv('TROTSKY_MODEL_API_KEY',''))
            reply=parse_reply({'response':text,'finish_reason':finish},request)
            session['state']=append_turn(state,body['question'],reply)
            return {'kind':'model_reply','reply':reply,'source_packets':request['source_packets'],'retrieval':request['retrieval'],
                'model_identity':'endpoint operator asserted; not remotely attested','omitted_history_pairs':request['omitted_history_pairs'],
                'token_budget_omitted_packets':request['token_budget_omitted_packets'],'usage':usage}
        finally:session['busy'].release()
    def feedback(self,body):
        if set(body)!={'case_id','turn','support','voice','usefulness','notes'} or type(body['turn']) is not int or not isinstance(body['notes'],str) or len(body['notes'])>4000:
            raise ValueError('Invalid review form')
        if body['support'] not in ('supported','needs_checking','unsupported') or body['voice'] not in ('convincing','mixed','unconvincing') or body['usefulness'] not in ('helpful','partial','missed_question'):
            raise ValueError('Choose review ratings')
        if not any(r['case_id']==body['case_id'] and r['turn']==body['turn'] for r in self.review_rows()):raise ValueError('Unknown reviewed sample')
        record={**body,'created_at':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),'reviewer':'user','review_status':'human_submitted'}
        self.feedback_path.parent.mkdir(parents=True,exist_ok=True)
        with self.lock,(self.feedback_path).open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return {'saved':True,'storage':'local_project_only'}


def handler_for(workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def origin_ok(self):
            hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            return (self.headers.get('Host') in hosts and (not origin or origin in {'http://'+h for h in hosts})
                and (self.command=='GET' or self.headers.get('Sec-Fetch-Site')!='cross-site'))
        def send(self,status,data,content_type='application/json; charset=utf-8'):
            raw=json.dumps(data,ensure_ascii=False).encode() if not isinstance(data,bytes) else data
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.origin_ok():self.send(403,{'error':'Local same-origin access required'});return
            try:
                if self.path=='/api/bootstrap':self.send(200,workbench.bootstrap())
                elif self.path=='/api/review':self.send(200,{'rows':workbench.review_rows()})
                elif self.path in ('/','/app.js','/style.css'):
                    name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[self.path]
                    kind={'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name]
                    self.send(200,(STATIC/name).read_bytes(),kind+'; charset=utf-8')
                else:self.send(404,{'error':'Not found'})
            except Exception:self.send(500,{'error':'Local review data could not be loaded; inspect the project logs'})
        def do_POST(self):
            if not self.origin_ok():self.send(403,{'error':'Local same-origin access required'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if self.headers.get('Content-Type','').split(';')[0]!='application/json' or not 0<length<=64000:raise ValueError('A bounded JSON request is required')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('Expected a JSON object')
                if self.path=='/api/feedback':self.send(200,workbench.feedback(body));return
                cookies=SimpleCookie()
                try:cookies.load(self.headers.get('Cookie',''))
                except CookieError:pass
                identity=cookies['trotsky_review'].value if 'trotsky_review' in cookies else ''
                if not identity or len(identity)>80 or any(ch not in '0123456789abcdef' for ch in identity):
                    identity=secrets.token_hex(24)
                if self.path=='/api/reset':
                    with workbench.lock:workbench.sessions.pop(identity,None)
                    self.send(200,{'reset':True});return
                if self.path not in ('/api/lookup','/api/chat'):self.send(404,{'error':'Not found'});return
                response=workbench.ask(identity,body,live=self.path=='/api/chat')
                # Session ID is local-only, never sent to an inference provider.
                raw=json.dumps(response,ensure_ascii=False).encode();self.send_response(200)
                self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)))
                self.send_header('Set-Cookie','trotsky_review='+identity+'; HttpOnly; SameSite=Strict; Path=/')
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
            except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':str(exc)})
            except Exception:self.send(502,{'error':'The model request failed. No automatic retry was made.'})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--snapshot',type=Path,default=ROOT/'evaluation/ghost_reviewed_coverage_v3')
    p.add_argument('--review-run',type=Path,default=ROOT/'evaluation/runs/ghost_review_chat_v3')
    p.add_argument('--feedback',type=Path,default=ROOT/'evaluation/review_workbench/feedback.jsonl')
    p.add_argument('--base-url');p.add_argument('--model',default=MODEL['model_id'],choices=['Qwen/Qwen3.5-9B','Qwen/Qwen3.5-27B'])
    p.add_argument('--tokenizer',type=Path);args=p.parse_args();tokenizer=None
    if args.base_url:
        if not args.tokenizer:p.error('A matching local tokenizer directory is required for live request budgets')
        from remote_model import chat_endpoint
        chat_endpoint(args.base_url)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    workbench=Workbench(args.snapshot,args.review_run,args.feedback,base_url=args.base_url,model=args.model,tokenizer=tokenizer)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(workbench))
    print('Review workbench: http://127.0.0.1:'+str(server.server_port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
