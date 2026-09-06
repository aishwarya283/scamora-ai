package com.example.scamora

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private var contactChannel: MethodChannel? = null
    private var contactService: ContactCheckService? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        contactChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "scamora/contact_check")
        contactService = ContactCheckService(this, contactChannel!!)
        contactService?.register()
    }
}
