window.mountChainlitWidget({
  chainlitServer: "http://localhost:8000",
***REMOVED******REMOVED***;

window.addEventListener("chainlit-call-fn", (e***REMOVED*** => {
  const { name, args, callback ***REMOVED*** = e.detail;
  if (name === "formfill"***REMOVED*** {
***REMOVED***console.log(name, args***REMOVED***;
***REMOVED***dash_clientside.set_props("fieldA", {value: args.fieldA***REMOVED******REMOVED***;
***REMOVED***dash_clientside.set_props("fieldB", {value: args.fieldB***REMOVED******REMOVED***;
***REMOVED***dash_clientside.set_props("fieldC", {value: args.fieldC***REMOVED******REMOVED***;
***REMOVED***callback("You sent: " + args.fieldA + " " + args.fieldB + " " + args.fieldC***REMOVED***;
  ***REMOVED***
***REMOVED******REMOVED***;





