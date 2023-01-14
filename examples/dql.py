from botdyn.symbol import Symbol, Expression
from botdyn.post_processors import StripPostProcessor
from botdyn.pre_processors import PreProcessor
import botdyn as bd


DQL_CONTEXT = """[Description]
The semantics of DQL are similar to SQL however the syntax is as follows:
- SQL: FROM, DQL: fetch, from:...
- SQL: SELECT, DQL: fields
- SQL: WHERE, DQL: filter
- SQL: LIMIT, DQL: limit
- SQL: and, DQL: |
// The following statement shows a typical DQL outcome of an iterative process of creating a DQL query. It used a few measures to speed up the process: The query timeframe was reduced to 'Last 1h'. Sampling was used to further speed up the response time and because the precision of the result was not required. The 'limit' command was used to reduce the number of records as only a few were needed to proceed. The 'fields' command was used to reduce the number of fields(columns) in the output to ease the readability of the query output
DQL: fetch logs, from:now()-1h, samplingRatio:100
| fieldsRename severity=loglevel, logfile=log.source, message = content, container=k8s.container.name
| fields timestamp, severity, logfile, message, container
| limit 100
| filterOut severity == “NONE” or severity == “INFO”
| filter contains(container,“dns”)
| filter matchesPhrase(message,“No files matching import”)
| summarize count(), by:severity
[Example]
// Get from table logs the entities dt.entity.process_group that equals to “PROCESS_GROUP-123F4A56BCDA0EA9”
SQL: SELECT logs.dt.entity.process_group FROM logs WHERE logs.dt.entity.process_group = “PROCESS_GROUP-123F4A56BCDA0EA9"
DQL: fetch logs | filter dt.entity.process_group==“PROCESS_GROUP-123F4A56BCDA0EA9”
[Example]
// Get from table logs the entities dt.entity.service_id that equals to “SERVICE_ID_1223844839"
SQL: SELECT from logs WHERE dt.entity.service_id = “SERVICE_ID_1223844839”
DQL: fetch logs | filter dt.entity.service_id = “SERVICE_ID_1223844839"
[Example]
// Example query to fetch all logs from the last hour until now. Filter files if they end by “pgi.log”, parse the results based on the content “LD IPADDR:ip ':' LONG:payload SPACE LD 'HTTP_STATUS' SPACE INT:http_status LD (EOL| EOS)“. Afterwards, summarize on the fields total_payload = sum(payload), failedRequests = countIf(http_status >= 400) and successfulRequests = countIf(http_status <= 400) by ip and host.name. Then sort in descendent order the field failedRequests.
DQL: fetch       logs, from:now()-10m
| filter    endsWith(log.source,“pgi.log”)
| parse     content, “LD IPADDR:ip ':' LONG:payload SPACE LD 'HTTP_STATUS' SPACE INT:http_status  LD (EOL| EOS)”
| summarize total_payload = sum(payload),
            failedRequests = countIf(http_status >= 400),
            successfulRequests = countIf(http_status <= 400),
            by:{ip, host.name}
| fieldsAdd total_payload_MB = total_payload/1000000
| fields    ip, host.name, failedRequests, successfulRequests, total_payload_MB
| sort      failedRequests desc
[Example]
// Sampling is non-deterministic, and will return a different result set with each query run. Also, all the following commands will work based on the sampled set of input data, yielding unprecise aggregates. {{/callout}}
DQL: fetch logs, from:now()-7d, samplingRatio:100
| summarize c = countIf(loglevel == “ERROR”), by:bin(timestamp, 3h)
| fieldsAdd c = c*100
[Example]
// In the following example, only three of all available fields returned from the fetch stage are selected. The loglevel field is additionally converted to lowercase by the DQL lower function.
DQL: fetch logs
| fields timestamp, severity = lower(loglevel), content
[Example]
// The following example shows the difference between fields and fieldsAdd. While the fields command defines the result table by the fields specified, the fieldsAdd command adds new fields to the existing fields.
DQL: fetch logs
| fieldsAdd severity = lower(loglevel)
[Task]
Generate a DQL based on the above descripiton and examples:"""


class DQLPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return '// {}\n DQL:'.format(str(wrp_self))


class DQL(Expression):
    @property
    def static_context(self):
        return DQL_CONTEXT + super().static_context
    
    def forward(self, sym: Symbol, *args, **kwargs):
        @bd.few_shot(prompt="Generate queries based on a domain specific language Dynatrace Query Language (DQL)\n", 
                     examples=[],
                     pre_processor=[DQLPreProcessor()],
                     post_processor=[StripPostProcessor()], **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(DQL(sym)))
    
    @property
    def _sym_return_type(self):
        return DQL
