"""Local OpenAI-compatible streaming server for native NTSA-X recurrent weights.

Compatible with live_gateway.py --use-student; *no* teacher model is loaded.
Strict loopback binding, bounded generation, serial model access, no public auth.
"""
from __future__ import annotations
from argparse import ArgumentParser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4
import json
import torch

from ntsa_x.native_model import ByteCodec, utf8_stream
from ntsa_x.native_train import load_checkpoint


def handler_factory(model,model_id,port,max_new_tokens):
    semaphore=BoundedSemaphore(1)
    codec=ByteCodec()

    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"

        def respond(self,status,item):
            data=json.dumps(item).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path=="/health":
                return self.respond(200,{"status":"ok","architecture":
                                          "ntsa-x-native-recurrent-v1",
                                          "teacher_loaded":False})
            return self.respond(404,{"error":"Not found"})

        def do_POST(self):
            if (self.path!="/v1/chat/completions"
                    or self.headers.get("Host") not in
                       (f"127.0.0.1:{port}",f"localhost:{port}")
                    or self.headers.get("Origin")):
                return self.respond(403,{"error":"Disallowed request"})
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":
                return self.respond(415,{"error":"Expected JSON"})
            try:
                size=int(self.headers.get("Content-Length","-1"))
                if not 1<=size<=32768:
                    raise ValueError("Invalid request size")
                query=json.loads(self.rfile.read(size))
                if not isinstance(query,dict):
                    raise ValueError("Expected object")
                messages=query["messages"]
                requested=query.get("model",model_id)
                if requested!=model_id:
                    raise ValueError("Model ID mismatch")
                tokens=int(query.get("max_tokens",max_new_tokens))
                if not 1<=tokens<=max_new_tokens:
                    raise ValueError("max_tokens exceeds bound")
                temperature=float(query.get("temperature",0))
                if not 0<=temperature<=2:
                    raise ValueError("Invalid temperature")
                if query.get("stream") is not True:
                    raise ValueError("Only stream=true is supported")
                # Reuse the exact prompt framing used during native training.
                prefix,_=codec.conversation(messages,generation=True,
                                            max_tokens=model.config.max_context)
            except (ValueError,TypeError,KeyError,UnicodeError) as exc:
                return self.respond(400,{"error":str(exc)})
            if not semaphore.acquire(blocking=False):
                return self.respond(503,{"error":"Native model busy"})
            self.close_connection=True
            try:
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream; charset=utf-8")
                self.send_header("Cache-Control","no-store")
                self.send_header("X-Content-Type-Options","nosniff")
                self.send_header("Connection","close")
                self.end_headers()
                trace=uuid4().hex
                for text in utf8_stream(model.generate(
                        prefix,max_new_tokens=tokens,temperature=temperature)):
                    payload={"id":trace,"object":"chat.completion.chunk",
                             "choices":[{"index":0,"delta":{"content":text},
                                         "finish_reason":None}]}
                    self.wfile.write(("data: "+json.dumps(payload,ensure_ascii=False)
                                      +"\n\n").encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,OSError):
                pass
            finally:
                semaphore.release()

        def log_message(self, *_args):
            pass  # Never log user messages into stdout.

    return Handler


def main():
    p=ArgumentParser(description="Local native NTSA-X recurrent streaming model")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--model-id",default="ntsa-native-v1")
    p.add_argument("--port",type=int,default=8081)
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    if not 1024<=args.port<=65535 or not 1<=args.max_new_tokens<=2048:
        p.error("Port or generation bound out of range")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    model=load_checkpoint(args.checkpoint,device=args.device)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),
               handler_factory(model,args.model_id,args.port,args.max_new_tokens))
    server.daemon_threads=True
    print(f"Native NTSA-X student running locally on 127.0.0.1:{args.port}",
          flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
