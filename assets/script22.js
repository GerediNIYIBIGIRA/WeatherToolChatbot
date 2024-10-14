window.mountChainlitWidget({
  chainlitServer:"http://localhost:8000",
});
console.log('script loaded and running');

window.addEventListener("chainlit-call-fn", (e) => {
  const { name, args, callback } = e.detail;
  if (name === "formfill") {
    console.log(name, args);
    console.log('Recieved data from chainlit:', args);

    document.getElementById("fieldA").value = args.fieldA;
    document.getElementById("fieldB").value = args.fieldB;
    const resultString = Array.isArray(args.fieldC) ? args.fieldC.map(item => item.text || JSON.stringify(item)).join(", ") : args.fieldC;
    document.getElementById("fieldC").value = resultString;

    callback("You sent: " + args.fieldA + " " + args.fieldB + " " + args.fieldC);
  }
});



