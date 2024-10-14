window.mountChainlitWidget({
  chainlitServer:"http://localhost:8000",
***REMOVED******REMOVED***;
console.log('script loaded and running'***REMOVED***;

window.addEventListener("chainlit-call-fn", (e***REMOVED*** => {
  const { name, args, callback ***REMOVED*** = e.detail;
  if (name === "formfill"***REMOVED*** {
***REMOVED***console.log(name, args***REMOVED***;
***REMOVED***console.log('Recieved data from chainlit:', args***REMOVED***;

***REMOVED***document.getElementById("fieldA"***REMOVED***.value = args.fieldA;
***REMOVED***document.getElementById("fieldB"***REMOVED***.value = args.fieldB;
***REMOVED***const resultString = Array.isArray(args.fieldC***REMOVED*** ? args.fieldC.map(item => item.text || JSON.stringify(item***REMOVED***.join(", "***REMOVED*** : args.fieldC;
***REMOVED***document.getElementById("fieldC"***REMOVED***.value = resultString;

***REMOVED***callback("You sent: " + args.fieldA + " " + args.fieldB + " " + args.fieldC***REMOVED***;
  ***REMOVED***
***REMOVED******REMOVED***;



